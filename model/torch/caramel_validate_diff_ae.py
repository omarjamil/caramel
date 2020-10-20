import torch
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py

import model as nn_model
import data_io


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_args(model_file, normaliser, data_region):
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    args = checkpoint['arguments']
    args.isambard = False
    args.region = data_region
    args.device = torch.device('cpu')
    args.normaliser = normaliser
    if args.isambard:
        args.locations={ "train_test_datadir":"/home/mo-ojamil/ML/CRM/data",
                "chkpnt_loc":"/home/mo-ojamil/ML/CRM/data/models/chkpts",
                "hist_loc":"/home/mo-ojamil/ML/CRM/data/models",
                "model_loc":"/home/mo-ojamil/ML/CRM/data/models/torch",
                "normaliser_loc":"/home/mo-ojamil/ML/CRM/data/normaliser/{0}".format(args.normaliser)}
    else:
        args.locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    # args.xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
    # args.yvars=['qtot_next', 'theta_next']
    # args.yvars2=['qphys', 'theta_phys']
    # args.train_on_y2 = True
    return args

def set_model(model_file, args):

    # Define the Model
    # n_inputs,n_outputs=140,70
    print(args.xvars)
    args.region=args.data_region
    # in_features = (args.nlevs*(len(args.xvars)-3)+3)
    in_features = (args.nlevs*(len(args.xvars)))
    print(in_features)
    nb_classes = (args.nlevs*(len(args.yvars)))
    # in_features, nb_classes=(args.nlevs*4+3),(args.nlevs*2)
    # hidden_size = 512
    # hidden_size = int(1. * in_features + nb_classes)
    # mlp = nn_model.MLP(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLPSkip(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    # mlp = nn_model.VAE(in_features)
    mlp = nn_model.AE(in_features)
    # mlp = nn_model.MLPDrop(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    # Load the save model 
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval() 
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    return mlp

def evaluate_qdiff(model, datasetfile, args):

    nn_data = data_io.Data_IO_validation_x2(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        xvars2=args.xvars2,
                        no_norm = False)
    
    x,y,y2,qxmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()

    y = x.data.clone()
    # model = set_model(args)
    # yp, mu, logvar = model(x)
    encoded, yp = model(x)
    # print(mu[0,:],logvar[0,:])
    hfilename = args.model_name.replace('.tar','_qnext.hdf5')

    output={'qtot':x.data.numpy(),
            'qtot_ml':yp.data.numpy(), 
        }
    
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,1,figsize=(14,10))
    ax = axs[0]
    c = ax.pcolor(x[:,:].data.numpy().T)
    ax.set_title('q')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(yp[:,:].data.numpy().T)
    ax.set_title('q ML')
    fig.colorbar(c,ax=ax)

    ax = axs[2]
    c = ax.pcolor((yp[:,:] - x[:,:]).data.numpy().T)
    ax.set_title('ML - Truth')
    fig.colorbar(c,ax=ax)
    
    plt.show()

if __name__ == "__main__":
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = model_loc+"qdiff_ae_normed_006_lyr_055_in_055_out_0110_hdn_050_epch_00150_btch_023001AQTS_mse_023001AQT_normalise_stkd.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_020.hdf5"
    normaliser_region = "023001AQT_normalise"
    # normaliser_region = "023001AQ_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    evaluate_qdiff(model, datasetfile, args)