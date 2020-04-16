import torch
import pickle
import numpy as np
import math
import sys
import os
import h5py
import argparse
import matplotlib.pyplot as plt

import model
import data_io

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args_parser = None
def set_args():
    if args_parser is not None:
        args = args_parser
    else:
        args = dotdict()
        args.seed = 1
        args.log_interval = 10    
        args.batch_size = 10000
        args.epochs = 1
        args.with_cuda = False
        args.chkpt_interval = 10
        args.isambard = False
        args.warm_start = False
        args.identifier = '021501AQ'
        args.data_region = '021501AQ'
        args.normaliser = 'standardise'
        args.loss = 'mae'
        args.nb_hidden_layers = 9
        args.nlevs = 45
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
    args.model_name = "q_qadv_t_tadv_swtoa_lhf_shf_qtphys_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_{7}_{8}levs.tar".format(str(args.nb_hidden_layers).zfill(3),
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

def train_dataloader(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    concatDataset = data_io.ConcatDataset("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], data_frac=args.data_fraction)
    train_loader = torch.utils.data.DataLoader(concatDataset,batch_size=args.batch_size, shuffle=True)
    return train_loader, concatDataset

def plot_training_data(args):
    train_ldr, concatDataset =  train_dataloader(args)
    for batch_idx, batch in enumerate(train_ldr):
        x,y = batch
        print(x.shape, y.shape)
        x_split = concatDataset.split_xdata(x)
        y_split = concatDataset.split_ydata(y)
        qphys_test_norm = y_split['qphys']
        theta_phys_test_norm = y_split['theta_phys']
        qtot_test_norm = x_split['qtot']
        qadv_test_norm = x_split['qadv']
        theta_test_norm = x_split['theta']
        theta_adv_test_norm = x_split['theta_adv']
        sw_toa = x_split['sw_toa']
        shf = x_split['shf']
        lhf = x_split['lhf']

        level = 0
        fig, axs = plt.subplots(3,3,figsize=(14, 10),sharex=True)
        ax = axs[0,0]
        ax.plot(qphys_test_norm[:,level],'.-',label='qphys')
        ax.legend()

        ax = axs[1,0]
        ax.plot(theta_phys_test_norm[:,level],'.-',label='tphys')
        ax.legend()
        
        ax = axs[2,0]
        ax.plot(qadv_test_norm[:,level],'.-',label='qadv')
        ax.legend()

        ax = axs[0,1]
        ax.plot(sw_toa[:],'.-',label='sw_toa')
        ax.legend()

        ax = axs[1,1]
        ax.plot(theta_adv_test_norm[:,level],'.-',label='tadv')
        ax.legend()
        
        ax = axs[2,1]
        ax.plot(theta_test_norm[:,level],'.-',label='theta')
        ax.legend()
        
        ax = axs[0,2]
        ax.plot(qtot_test_norm[:,level],'.-',label='qtot')
        ax.legend()

        ax = axs[1,2]
        ax.plot(shf[:],'.-',label='shf')
        ax.legend()

        ax = axs[2,2]
        ax.plot(lhf[:],'.-',label='lhf')
        ax.legend()

        ax.set_title('Level {0}'.format(level))
        ax.legend()
        plt.show()

if __name__ == "__main__":
    args = set_args()
    plot_training_data(args)
    