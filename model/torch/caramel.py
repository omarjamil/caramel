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

def minkowski_error(prediction, target, minkowski_parameter=1.5):
    """
    Minkowski error to be better deal with outlier errors
    """
    loss = torch.mean((torch.abs(prediction - target))**minkowski_parameter)
    return loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# args = dotdict()

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
parser.add_argument('--nhdn-layers', type=int, default=6, metavar='N',
                    help='Number of hidden layers (default: 6)')
    
args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def configure_optimizers():
    optimizer =  torch.optim.Adam(mlp.parameters(), lr=1.e-3)
    # optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # loss_function = torch.nn.MSELoss()
    return optimizer, scheduler

def train_loader():
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(locations["train_test_datadir"],region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("train",nlevs,train_dataset_file, overfit=False),
             batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

def test_loader():
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(locations["train_test_datadir"],region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("test",nlevs, test_dataset_file, overfit=False),
             batch_size=batch_size, shuffle=False, **kwargs)
    return validation_loader

def training_step(batch, batch_idx):
    """
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    output = mlp(x)
    loss = loss_function(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_idx):
    """
    """
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        output = mlp(x)
        loss = loss_function(output, y)
    return loss


def checkpoint_save(epoch: int, nn_model: model, nn_optimizer: torch.optim, training_loss: list, validation_loss: list, model_name: str):
    """
    Save model checkpoints
    """
    checkpoint_name = model_name.replace('.tar','_chkepo_{0}.tar'.format(str(epoch).zfill(3)))
    torch.save({'epoch':epoch,
                'model_state_dict':nn_model.state_dict(),
                'optimizer_state_dict':nn_optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss},
                locations['model_loc']+'/'+checkpoint_name)


def set_model():
    global mlp
    global loss_function
    global optimizer
    global scheduler
    
    mlp = model.MLP(in_features, nb_classes, nb_hidden_layer, hidden_size)
    # mlp = model.MLP_BN(in_features, nb_classes, nb_hidden_layer, hidden_size)
    pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Number of traninable parameter: {0}".format(pytorch_total_params))
    loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss()
    # loss_function = minkowski_error
    optimizer, scheduler = configure_optimizers()

    if args.warm_start:
    # Load the save model 
        checkpoint = torch.load(locations['model_loc']+'/'+model_name, map_location=device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        model_name = model_name.replace('.tar','_{0}.tar'.format(region))
    else:
        mlp.to(device)
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())

def train_loader():
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(locations["train_test_datadir"],region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("train",nlevs,train_dataset_file, overfit=True),
             batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

def test_loader():
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(locations["train_test_datadir"],region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("test",nlevs, test_dataset_file, overfit=True),
             batch_size=batch_size, shuffle=False, **kwargs)
    return validation_loader

def set_args():

    global kwargs
    global log_interval
    global region
    global epochs
    global batch_size
    global nb_hidden_layer
    global identifier
    global nlevs
    global in_features
    global nb_classes
    global hidden_size
    global model_name
    global locations

    kwargs = {'pin_memory': False} if args.cuda else {}

    log_interval = args.log_interval

    # Define the Model
    # n_inputs,n_outputs=140,70
    region=args.data_region
    epochs=args.epochs
    batch_size=args.batch_size
    nb_hidden_layer = args.nhdn_layers
    identifier = args.identifier
    nlevs = 45
    in_features = (nlevs*4+3)
    nb_classes =(nlevs*2)
    hidden_size = 256

    model_name = "q_qadv_t_tadv_swtoa_lhf_shf_qtphys_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_mae_vlr.tar".format(str(nb_hidden_layer).zfill(3),
                                                                                    str(in_features).zfill(3),
                                                                                    str(nb_classes).zfill(3),
                                                                                    str(hidden_size).zfill(4),
                                                                                    str(epochs).zfill(3),
                                                                                    str(batch_size).zfill(5),
                                                                                    identifier)

    # Get the data
    if args.isambard:
        locations={ "train_test_datadir":"/home/mo-ojamil/ML/CRM/data",
                "chkpnt_loc":"/home/mo-ojamil/ML/CRM/data/models/chkpts",
                "hist_loc":"/home/mo-ojamil/ML/CRM/data/models",
                "model_loc":"/home/mo-ojamil/ML/CRM/data/models/torch",
                "normaliser_loc":"/home/mo-ojamil/ML/CRM/data/normaliser/{0}".format(region)}
    else:
        locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(region)}

def train_loop():
    
    training_loss = []
    train_ldr = train_loader()
    validation_loss = []
    test_ldr = test_loader()

    for epoch in range(1, epochs + 1):
        ## Training
        train_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            # Sets the model into training mode
            mlp.train()
            loss = training_step(batch, batch_idx)
            train_loss += loss
            if batch_idx % log_interval == 0:
                x,y=batch
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
            mlp.eval()
            loss = validation_step(batch, batch_idx)
            test_loss += loss.item()
        average_loss_val = test_loss / len(test_ldr.dataset)
        print('====> validation loss: {:.6f}'.format(average_loss_val))
        validation_loss.append(average_loss_val)
        # if epoch % 5 == 0:
        #     checkpoint_save(epoch, mlp, optimizer, training_loss, validation_loss, model_name)            
     # Save the final model
    torch.save({'epoch':epoch,
                'model_state_dict':mlp.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss},
                locations['model_loc']+'/'+model_name)   


if __name__ == "__main__":
    set_args()
    set_model()
    train_loop()
    
 