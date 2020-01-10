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
import normalize_tensors as nt


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

parser = argparse.ArgumentParser(description='Train Q')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--chkpt-interval', type=int, default=10, metavar='N',
                    help='how many epochs before saving a checkpoint')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

# Data normalisers
# qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
# q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
# qadd_normaliser = joblib.load('{0}/minmax_qadd_dot.joblib'.format(locations['normaliser_loc']))
qphys_normaliser = h5py.File('{0}/minmax_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
q_normaliser = h5py.File('{0}/minmax_qtot.hdf5'.format(locations['normaliser_loc']),'r')
qadd_normaliser = h5py.File('{0}/minmax_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
# Define the Model
n_inputs,n_outputs=140,70
mlp = model.MLP(n_inputs,n_outputs)
optimizer =  torch.optim.Adam(mlp.parameters())
loss_function = torch.nn.MSELoss()

# Get the data
region="50S69W"

train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
q_norm_train = train_data_in["qtot"]
qnext_norm_train = train_data_in["qtot_next"]
qadv_norm_train = train_data_in["qadv"]
qadv_dot_norm_train = train_data_in["qadv_dot"]
qphys_norm_train = train_data_out["qphys_tot"]

q_norm_test = test_data_in["qtot_test"]
qnext_norm_test = test_data_in["qtot_next_test"]
qadv_norm_test = test_data_in["qadv_test"]
qadv_dot_norm_test = test_data_in["qadv_dot_test"]
qphys_norm_test = test_data_out["qphys_test"]
qadd_train = train_data_in["qadd"]
qadd_dot_train = train_data_in["qadd_dot"]
qadd_test = test_data_in["qadd_test"]
qadd_dot_test = test_data_in["qadd_dot_test"]
qcomb_train = np.concatenate((qadv_norm_train,q_norm_train),axis=1)
qcomb_test  = np.concatenate((qadv_norm_test,q_norm_test),axis=1)
# qcomb_dot_train = np.concatenate((qadv_dot_norm_train,q_norm_train),axis=1)
# qcomb_dot_test  = np.concatenate((qadv_dot_norm_test,q_norm_test),axis=1)
qcomb_dot_train = np.concatenate((qadd_dot_train,q_norm_train),axis=1)
qcomb_dot_test  = np.concatenate((qadd_dot_test,q_norm_test),axis=1)

#train_in, train_out = qcomb_dot_train, qphys_norm_train
#test_in, test_out = qcomb_dot_test, qphys_norm_test
x,y,z = qcomb_dot_train, qphys_norm_train, qnext_norm_train
x_t,y_t = qcomb_dot_test, qphys_norm_test

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(x,y,z),
             batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
             ConcatDataset(x_t,y_t),
             batch_size=args.batch_size, shuffle=False)

def q_loss(qphys_prediction, qnext, qin):
    """
    Extra loss for q predicted from the model
    """
    global qphys_normaliser
    global q_normaliser
    global qadd_normaliser
    qadd_dot = qin.data.numpy()[:,:70]
    qadd_dot_denorm = qadd_normaliser.inverse_transform(qadd_dot)
    qphys_prediction_denorm = qphys_normaliser.inverse_transform(qphys_prediction.data.numpy())
    qnext_calc = qadd_dot_denorm + qphys_prediction_denorm
    qnext_calc_norm = q_normaliser.transform(qnext_calc)
    qnext_calc_norm_tensor = torch.from_numpy(qnext_calc_norm)
    loss = loss_function(qnext_calc_norm_tensor, qnext)
    return loss

def q_loss_tensors(qphys_prediction, qnext, qin):
    """
    Extra loss for q predicted from the model
    """
    global qphys_normaliser
    global q_normaliser
    global qadd_normaliser
    qadd_dot = qin.data[:,:70]
    qadd_dot_denorm = nt.inverse_minmax_tensor(qadd_normaliser, qadd_dot)
    qphys_prediction_denorm = nt.inverse_minmax_tensor(qphys_normaliser, qphys_prediction)
    qnext_calc = qadd_dot_denorm + qphys_prediction_denorm
    qnext_calc_norm = nt.minmax_tensor(q_normaliser, qnext_calc)
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
        qnext_loss = q_loss_tensors(prediction, z, x)
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
    model_name = "qcomb_add_dot_qloss_qphys_deep_test.tar"
    torch.save({'epoch':epoch,
                'model_state_dict':mlp.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':training_loss},
                locations['model_loc']+'/'+model_name)

    # Save model training history
    # history={'training_loss':training_loss,'testing_loss':testing_loss}
    # hfilename = '{0}/{1}'.format(locations['hist_loc'],'q_history.h5')
    # with h5py.File(hfilename, 'w') as hfile:
    #     for k, v in history.items():  
    #         hfile.create_dataset(k,data=v)    