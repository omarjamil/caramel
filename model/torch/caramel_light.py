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
import pytorch_lightning as pl
from torch.nn import functional as F

class MLP(pl.LightningModule):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
        # super(MLP, self).__init__()
        super().__init__()
        # self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = F.relu(l(x))
        x = self.out(x)
        return x

    # def loss_function(self, prediction, target, minkowski_parameter=1.5):
    #     """
    #     Minkowski error to be better deal with outlier errors
    #     """
    #     loss = torch.mean((torch.abs(prediction - target))**minkowski_parameter)
    #     return loss
    
    # def loss_function(self, prediction, target):
    #     """
    #     MSE
    #     """
    #     return torch.nn.MSELoss(prediction, target)
    
    def loss_function(self, prediction, target):
        """
        L1 Error
        """
        return torch.nn.L1Loss(prediction, target)

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr=1.e-3)
        # optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        # loss_function = torch.nn.MSELoss()
        return optimizer, scheduler

    def train_loader(self):
        train_dataset_file = "{0}/train_data_{1}.hdf5".format(locations["train_test_datadir"],region)
        train_loader = torch.utils.data.DataLoader(
                data_io.ConcatDataset("train",nlevs,train_dataset_file, overfit=False),
                batch_size=batch_size, shuffle=True)
        return train_loader

    def test_loader(self):
        test_dataset_file = "{0}/test_data_{1}.hdf5".format(locations["train_test_datadir"],region)
        validation_loader = torch.utils.data.DataLoader(
                data_io.ConcatDataset("test",nlevs, test_dataset_file, overfit=False),
                batch_size=batch_size, shuffle=False)
        return validation_loader

    def training_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output,y)
        logs = {'train_loss': loss}
        return {'loss':loss,'log':logs}

    def validation_step(self, batch, batch_idx):
        """
        """
        x,y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        return {'val_loss':loss}

    # def validation_epoch_end(self, outputs):
    #     # called at the end of the validation epoch
    #     # outputs is an array with what you returned in validation_step for each batch
    #     # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


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
# device = torch.device("cuda" if args.cuda else "cpu")



region=args.data_region
epochs=args.epochs
batch_size=args.batch_size
nb_hidden_layer = args.nhdn_layers
identifier = args.identifier
nlevs = 45
in_features = (nlevs*4+3)
nb_classes =(nlevs*2)
hidden_size = 512

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

# train
model = MLP(in_features, nb_classes, nb_hidden_layer, hidden_size)
trainer = pl.Trainer(max_epochs=epochs)
trainer.fit(model) 