import torch
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py
import argparse

import model
import data_io
import normalize

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
    
args = parser.parse_args()
args.cuda = args.with_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Define the Model
# n_inputs,n_outputs=140,70
n_inputs,n_outputs=70,70
n_layers = 6
mlp = model.MLP_06(n_inputs,n_outputs)
mlp.to(device)
optimizer =  torch.optim.Adam(mlp.parameters())
loss_function = torch.nn.MSELoss()

# Get the data
region="50S69W"

if args.isambard:
    locations={ "train_test_datadir":"/home/mo-ojamil/ML/CRM/data",
            "chkpnt_loc":"/home/mo-ojamil/ML/CRM/data/models/chkpts",
            "hist_loc":"/home/mo-ojamil/ML/CRM/data/models",
            "model_loc":"/home/mo-ojamil/ML/CRM/data/models/torch",
            "normaliser_loc":"/home/mo-ojamil/ML/CRM/data/normaliser"}
else:
    locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
            "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

# Data normalizer class
nt = normalize.Normalizers(locations)
# Training and testing data class
nn_data = data_io.Data_IO(region, locations)

# qcomb_train = np.concatenate((nn_data.qadv_norm_train,nn_data.q_norm_train),axis=1)
# qcomb_test  = np.concatenate((nn_data.qadv_norm_test,nn_data.q_norm_test),axis=1)

# comb_dot_train = np.concatenate((nn_data.qadd_dot_train,nn_data.q_norm_train),axis=1)
# comb_dot_test  = np.concatenate((nn_data.qadd_dot_test,nn_data.q_norm_test),axis=1)
qcomb_dot_train = nn_data.qadd_dot_train
qcomb_dot_test  = nn_data.qadd_dot_test

#train_in, train_out = qcomb_dot_train, qphys_norm_train
#test_in, test_out = qcomb_dot_test, qphys_norm_test
x,y,z = torch.from_numpy(qcomb_dot_train[:]).to(device), torch.from_numpy(nn_data.qphys_norm_train[:]).to(device), torch.from_numpy(nn_data.qnext_norm_train[:]).to(device)
x_t,y_t = torch.from_numpy(qcomb_dot_test[:]).to(device), torch.from_numpy(nn_data.qphys_norm_test[:]).to(device)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(x,y,z),
             batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
             ConcatDataset(x_t,y_t),
             batch_size=args.batch_size, shuffle=False)

def q_loss_tensors_mm(qphys_prediction, qnext, qin):
    """
    Extra loss for q predicted from the model
    """
    qadd_dot = qin.data[:,:70]
    qadd_dot_denorm = nt.inverse_minmax(qadd_dot, nt.qadd_mmscale, nt.qadd_feature_min, nt.qadd_feature_max, nt.qadd_data_min)
    qphys_prediction_denorm = nt.inverse_minmax(qphys_prediction, nt.qphys_mmscale, nt.qphys_feature_min, nt.qphys_feature_max, nt.qphys_data_min)
    qnext_calc = qadd_dot_denorm + qphys_prediction_denorm
    qnext_calc_norm = nt.minmax(qnext_calc, nt.q_mmscale, nt.q_feature_min, nt.q_feature_max, nt.q_data_min)
    loss = loss_function(qnext_calc_norm, qnext)
    return loss

def q_loss_tensors_std(qphys_prediction, qnext, qin):
    """
    Extra loss for q predicted from the model
    """
    qadd_dot = qin.data[:,:70]
    qadd_dot_denorm = nt.inverse_std(qadd_dot, (nt.qadd_stdscale).to(device), (nt.qadd_mean).to(device))
    qphys_prediction_denorm = nt.inverse_std(qphys_prediction, (nt.qphys_stdscale).to(device), (nt.qphys_mean).to(device))
    qnext_calc = qadd_dot_denorm + qphys_prediction_denorm
    qnext_calc_norm = nt.std(qnext_calc, (nt.q_stdscale).to(device), (nt.q_mean).to(device))
    loss = loss_function(qnext_calc_norm, qnext)
    return loss

def train(epoch):
    """
    Train the model
    """
     
    # Sets the model into training mode
    mlp.train()
    train_loss = 0
    # x=data in, y=data out, z = extra loss qnext
    for batch_idx, (x, y, z) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        prediction = mlp(x)
        loss = loss_function(prediction, y)
        # qnext_loss = q_loss_tensors_mm(prediction, z, x)
        qnext_loss = q_loss_tensors_std(prediction, z, x)
        loss += qnext_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, 
            batch_idx * len(x), len(train_loader.dataset),100. * batch_idx / len(train_loader),
            loss.item() / len(x)))
    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, average_loss))
    
    return average_loss

def test(epoch):
    mlp.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
            x = x.to(device)
            prediction = mlp(x)
            test_loss += loss_function(prediction,y).item()
            # if i == 0:
            #     n = min(x.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.6f}'.format(test_loss))
    return test_loss

if __name__ == "__main__":
    

    training_loss = []
    testing_loss = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        training_loss.append(train_loss)
        testing_loss.append(test_loss)
    # Save the final model
    model_name = "qloss_qphys_{1}deep_epoch_{0}_qadd_std.tar".format(str(args.epochs).zfill(3), str(n_layers).zfill(2))
    torch.save({'epoch':epoch,
                'model_state_dict':mlp.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':training_loss},
                locations['model_loc']+'/'+model_name)
 