import torch
import pickle
import numpy as np
import math
import sys
import os
import h5py
import argparse

import model
import data_io

parser = argparse.ArgumentParser(description='Train Q')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--with-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--chkpt-interval', type=int, default=10, metavar='N',
                    help='how many epochs before saving a checkpoint')
parser.add_argument('--isambard', action='store_true', default=False,
                    help='Run on Isambard GPU')
parser.add_argument('--warm-start', action='store_true', default=False,
                    help='Continue training')
parser.add_argument('--identifier', type=str, 
                    help='Added to model name as a unique identifier;  also needed for warm start from a previous model')                    
parser.add_argument('--data-region', type=str, help='data region')
parser.add_argument('--loss', type=str, help='loss function to use', default='mae')
parser.add_argument('--nb-hidden-layers', type=int, default=6, metavar='N',
                    help='Number of hidden layers (default: 6)')
parser.add_argument('--data-fraction', type=float, default=1.,
                    help='fraction of data to use for training and testing (default: 1)')      
parser.add_argument('--normaliser', type=str, 
                    help='Normalisation to use: standardise or normalise')
parser.add_argument('--nlevs', type=int, default=45, metavar='N',
                    help='Number of vertical levels to user')                                                      
args_parser = parser.parse_args()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def minkowski_error(prediction, target, minkowski_parameter=1.5):
    """
    Minkowski error to be better deal with outlier errors
    """
    loss = torch.mean((torch.abs(prediction - target))**minkowski_parameter)
    return loss

def configure_optimizers(model):
    optimizer =  torch.optim.Adam(model.parameters(), lr=1.e-3)
    # optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # loss_function = torch.nn.MSELoss()
    return optimizer, scheduler


def training_step(batch, batch_idx, model, loss_function, optimizer, device):
    """
    """
    x, y, y2 = batch
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    loss = loss_function(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_idx, model, loss_function, device):
    """
    """
    x,y, y2 = batch
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        output = model(x)
        loss = loss_function(output, y)
    return loss


def checkpoint_save(epoch: int, nn_model: model, nn_optimizer: torch.optim, training_loss: list, validation_loss: list, model_name: str, locations: dict, args):
    """
    Save model checkpoints
    """
    checkpoint_name = model_name.replace('.tar','_chkepo_{0}.tar'.format(str(epoch).zfill(3)))
    torch.save({'epoch':epoch,
                'model_state_dict':nn_model.state_dict(),
                'optimizer_state_dict':nn_optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                locations['model_loc']+'/'+checkpoint_name)


def set_model(args):
    mlp = model.MLP(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLP_BN(in_features, nb_classes, nb_hidden_layer, hidden_size)
    pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Number of traninable parameter: {0}".format(pytorch_total_params))

    if args.loss == "mae":
        loss_function = torch.nn.functional.l1_loss #torch.nn.L1Loss()
    elif args.loss == "mse":
        loss_function = torch.nn.functional.mse_loss #torch.nn.MSELoss()
    elif args.loss == "mink":
        loss_function = minkowski_error
    optimizer, scheduler = configure_optimizers(mlp)

    if args.warm_start:
    # Load the save model 
        checkpoint = torch.load(args.locations['model_loc']+'/'+args.model_name, map_location=args.device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        args.model_name = args.model_name.replace('.tar','_{0}.tar'.format(args.region))
    else:
        mlp.to(args.device)
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    
    return mlp, loss_function, optimizer, scheduler

def train_dataloader(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=True)
    return train_loader

def test_dataloader(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader



def train_loop(model, loss_function, optimizer, scheduler, args):
    
    training_loss = []
    train_ldr = train_dataloader(args)
    validation_loss = []
    test_ldr = test_dataloader(args)
    
    for epoch in range(1, args.epochs + 1):
        ## Training
        train_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            # Sets the model into training mode
            model.train()
            loss = training_step(batch, batch_idx, model, loss_function, optimizer, args.device)
            train_loss += loss
            if batch_idx % args.log_interval == 0:
                x,y, y2=batch
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, 
                batch_idx * len(x), len(train_ldr.dataset),100. * batch_idx / len(train_ldr),
                loss.item() / len(x)))
        average_loss = train_loss / len(train_ldr.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, average_loss))
        training_loss.append(average_loss)
        scheduler.step()
        ## Testing
        test_loss = 0
        for batch_idx, batch in enumerate(test_ldr):
            model.eval()
            loss = validation_step(batch, batch_idx, model, loss_function, args.device)
            test_loss += loss.item()
        average_loss_val = test_loss / len(test_ldr.dataset)
        print('====> validation loss: {:.6f}'.format(average_loss_val))
        validation_loss.append(average_loss_val)
        if epoch % 10 == 0:
            checkpoint_save(epoch, model, optimizer, training_loss, validation_loss, args.model_name, args.locations, args)            
     # Save the final model
    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                args.locations['model_loc']+'/'+args.model_name)   

def set_args():
    if args_parser is not None:
        args = args_parser
    else:
        args = dotdict()
        args.seed = 1
        args.log_interval = 10    
        args.batch_size = 100
        args.epochs = 1
        args.with_cuda = False
        args.chkpt_interval = 10
        args.isambard = False
        args.warm_start = False
        args.identifier = '021501AQ1H'
        args.data_region = '021501AQ1H'
        args.normaliser = 'standardise_mx'
        args.loss = 'mae'
        args.nb_hidden_layers = 9
        args.nlevs = 30
        args.data_fraction = 0.01

    torch.manual_seed(args.seed)
    args.cuda = args.with_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'pin_memory': False} if args.cuda else {}

    # Define the Model
    # n_inputs,n_outputs=140,70
    args.region=args.data_region
    args.in_features = (args.nlevs*4+3)
    args.nb_classes =(args.nlevs*2)
    args.hidden_size = int(0.66 * args.in_features + args.nb_classes)
    args.model_name = "q_qadv_t_tadv_swtoa_lhf_shf_qtnext_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_{7}_{8}_levs.tar".format(str(args.nb_hidden_layers).zfill(3),
                                                                                    str(args.in_features).zfill(3),
                                                                                    str(args.nb_classes).zfill(3),
                                                                                    str(args.hidden_size).zfill(4),
                                                                                    str(args.epochs).zfill(3),
                                                                                    str(args.batch_size).zfill(5),
                                                                                    args.identifier, 
                                                                                    args.loss,
                                                                                    args.normaliser)

    # Get the data
    if args.isambard:
        args.locations={ "train_test_datadir":"/home/mo-ojamil/ML/CRM/data",
                "chkpnt_loc":"/home/mo-ojamil/ML/CRM/data/models/chkpts",
                "hist_loc":"/home/mo-ojamil/ML/CRM/data/models",
                "model_loc":"/home/mo-ojamil/ML/CRM/data/models/torch",
                "normaliser_loc":"/home/mo-ojamil/ML/CRM/data/normaliser/{0}_{1}".format(args.region, args.normaliser)}
    else:
        args.locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}_{1}".format(args.region, args.normaliser)}
    return args

if __name__ == "__main__":
    args = set_args()
    model, loss_function, optimizer, scheduler = set_model(args)
    train_loop(model, loss_function, optimizer, scheduler, args)
    
 