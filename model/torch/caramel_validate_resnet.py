import torch
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py

import model
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
    # mlp = model.resnet18(args.nb_classes, args.in_channels)
    # mlp = model.ConvNet(args.in_channels, args.nlevs, args.nb_classes)
    mlp = model.ConvNNet(args.in_channels, args.nb_classes)
    # Load the save model 
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval() 
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    return mlp

def test_dataloader(args):
    test_dataset_file = "{0}/cnn_test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDatasetCNN2D("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             xvars2=args.xvars2, yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader

def evaluate_qnext(model, datasetfile, args):
 
    args.data_fraction = 0.01
    args.batch_size = 10
    test_ldr = test_dataloader(args)
    predictions = np.zeros([1, args.nlevs])
    truth = np.zeros([1,args.nlevs])
    persistence = np.zeros([1,args.nlevs])
    for batch_idx, batch in enumerate(test_ldr):
        print("Processing batch {0}".format(batch_idx))
        model.eval()
        x, y2, y, y2 = batch
        x = x.to(args.device)
        if args.train_on_y2:
            y = y2.to(args.device)
        else:
            y = y.to(args.device)
        with torch.no_grad():
            output = model(x)
            predictions = np.concatenate([predictions, output.data.numpy()[:]])
            truth = np.concatenate([truth, y.data.numpy()[:]])
            persistence = np.concatenate([persistence, x.data.numpy()[:,0,-1,:]])
        

    hfilename = args.model_name.replace('.tar','_qnext.hdf5')

    output={'qtotn_predict':np.array(predictions),
            'qtotn_test':np.array(truth), 
            'qtot':np.array(persistence)
            }
    
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)



if __name__ == "__main__":
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = model_loc+"qnext_004_in_052_out_040_epch_00500_btch_023001AQTT3T19_mae_023001AQ_standardise_mx_dfrac_1p0_3x5convnnet.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_015.hdf5"
    normaliser_region = "163001AQT_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    # evaluate_qtphys(model, datasetfile, args)
    # evaluate_add_adv(model, datasetfile, args)
    evaluate_qnext(model, datasetfile, args)
    # evaluate_qt_next(model, datasetfile, args)