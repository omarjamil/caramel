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
import normalize

def minkowski_error(prediction, target, minkowski_parameter=1.5):
    """
    Minkowski error to be better deal with outlier errors
    """
    loss = torch.mean((torch.abs(prediction - target))**minkowski_parameter)
    return loss

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
use_cuda = args.with_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
log_interval = args.log_interval

# Define the Model
# n_inputs,n_outputs=140,70
region=args.data_region
epochs=args.epochs
batch_size=args.batch_size
nb_hidden_layer = args.nhdn_layers
identifier = args.identifier
nlevs = 45
in_features, nb_classes=(nlevs*4+3),(nlevs*2)
hidden_size = 512
mlp = model.MLP(in_features, nb_classes, nb_hidden_layer, hidden_size)
# mlp = model.MLP_BN(in_features, nb_classes, nb_hidden_layer, hidden_size)
pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print("Number of traninable parameter: {0}".format(pytorch_total_params))

model_name = "q_qadv_t_tadv_swtoa_lhf_shf_qtphys_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_mae_vlr.tar".format(str(nb_hidden_layer).zfill(3),
                                                                                    str(in_features).zfill(3),
                                                                                    str(nb_classes).zfill(3),
                                                                                    str(hidden_size).zfill(4),
                                                                                    str(epochs).zfill(3),
                                                                                    str(batch_size).zfill(5),
                                                                                    identifier)
optimizer =  torch.optim.Adam(mlp.parameters(), lr=1.e-6)
# optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.L1Loss()
# loss_function = minkowski_error
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




# Data normalizer class
nt = normalize.Normalizers(locations)
# Training and testing data class
nn_data = data_io.Data_IO(region, locations)

x2d  = [nn_data.sw_toa_train, nn_data.lhf_train, nn_data.shf_train]
x2d_test  = [nn_data.sw_toa_test, nn_data.lhf_test, nn_data.shf_test]
x3d  = [nn_data.q_tot_train, nn_data.q_tot_adv_train, nn_data.theta_train, nn_data.theta_adv_train]
x3d_test  = [nn_data.q_tot_test, nn_data.q_tot_adv_test, nn_data.theta_test, nn_data.theta_adv_test]
y = [nn_data.qphys_train, nn_data.theta_phys_train]
y_test = [nn_data.qphys_test, nn_data.theta_phys_test]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, x3data, x2data, ydata, nlevs):
        self.x3datasets = x3data
        self.x2datasets = x2data
        self.ydatasets = ydata
        self.nlevs = nlevs

    def __getitem__(self, i):
        # x3 = [torch.tensor(d[i,:self.nlevs]) for d in self.x3datasets]
        # x2 = [torch.tensor(d[i]) for d in self.x2datasets]
        # x = torch.cat(x3+x2,dim=0)
        # y = torch.cat([torch.tensor(d[i,:self.nlevs]) for d in self.ydatasets],dim=0)
        x3 = [d[i,:self.nlevs] for d in self.x3datasets]
        x2 = [d[i] for d in self.x2datasets]
        x =  np.concatenate((x3+x2))
        y = np.concatenate(([d[i,:self.nlevs] for d in self.ydatasets]))

        return (torch.from_numpy(x).to(device),torch.from_numpy(y).to(device))

    def __len__(self):
        return min(len(d) for d in self.x2datasets)

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(x3d,x2d,y,nlevs),
             batch_size=batch_size, shuffle=True, **kwargs)

validate_loader = torch.utils.data.DataLoader(
             ConcatDataset(x3d_test,x2d_test,y_test,nlevs),
             batch_size=batch_size, shuffle=False, **kwargs)


def train(epoch):
    """
    Train the model
    """
     
    # Sets the model into training mode
    mlp.train()
    train_loss = 0
    # x=data in, y=data out, z = extra loss qnext
    # first_batch = next(iter(train_loader))
    # for batch_idx, (x, y) in enumerate([first_batch] * 50):
    # training code here
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        prediction = mlp(x)
        phys_loss = loss_function(prediction, y)
        # qnext_loss = q_loss_tensors_mm(prediction, z, x)
        # qnext_loss = q_loss_tensors_std(prediction, z, x)
        loss = phys_loss #+ 0.*qnext_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, 
            batch_idx * len(x), len(train_loader.dataset),100. * batch_idx / len(train_loader),
            loss.item() / len(x)))
        
        ##### Use this for finding learning rate 
        # scheduler.step()
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'], loss.item() / len(x))
        ###### LR finder end
    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, average_loss))
    
    return average_loss

def validate(epoch):
    mlp.eval()
    validation_loss = 0
    with torch.no_grad():
        # first_batch = next(iter(validate_loader))
        # for batch_idx, (x, y) in enumerate([first_batch] * 50):
        for batch_idx, (x,y) in enumerate(validate_loader):
            x = x.to(device)
            prediction = mlp(x)
            validation_loss += loss_function(prediction,y).item()
            # if i == 0:
            #     n = min(x.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    validation_loss /= len(validate_loader.dataset)
    print('====> validation loss: {:.6f}'.format(validation_loss))
    return validation_loss

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

def train_main():
    training_loss = []
    validation_loss = []
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        validate_loss = validate(epoch)
        scheduler.step()    
        training_loss.append(train_loss)
        validation_loss.append(validate_loss)
        if epoch % 5 == 0:
            checkpoint_save(epoch, mlp, optimizer, training_loss, validation_loss, model_name)

    # Save the final model
    torch.save({'epoch':epoch,
                'model_state_dict':mlp.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss},
                locations['model_loc']+'/'+model_name)

if __name__ == "__main__":
    train_main()
    
 