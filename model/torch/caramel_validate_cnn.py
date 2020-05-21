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
    in_features = (args.nlevs*(len(args.xvars)-3)+3)
    print(in_features)
    mlp = model.ConvNN(args.in_channels, args.nlevs, args.nb_classes)
    # Load the save model 
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval() 
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    return mlp



def evaluate_qnext(model, datasetfile, args):
    nn_data = data_io.Data_IO_validation_CNN(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'], 
                        xvars=args.xvars,
                        xvars2=args.xvars2,
                        yvars=args.yvars,
                        yvars2=args.yvars2)
    
    x,x2,y,y2,xmean,xstd,xmean2,xstd2,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    yp = model(x)
    qtot_denorm = nn_data._inverse_transform(x[:,0,:], xmean[0,:], xstd[0,:])
    qadv_inv = nn_data._inverse_transform(x[:,1,:], xmean[1,:], xstd[1,:])
    theta_denorm = nn_data._inverse_transform(x[:,2,:], xmean[2,:], xstd[2,:])
    theta_adv_inv = nn_data._inverse_transform(x[:,3,:], xmean[3,:], xstd[3,:])

    yinv = nn_data._inverse_transform(y,ymean,ystd)
    ypinv = nn_data._inverse_transform(yp,ymean,ystd)

    qtotn_test_denorm = yinv[:,:args.nlevs]
    qtotn_predict_denorm = ypinv[:,:args.nlevs]
    
    qtotn_test_norm = y[:,:args.nlevs]
    qtotn_predict_norm = yp[:,:args.nlevs]

    hfilename = args.model_name.replace('.tar','_qnext.hdf5')

    output={'qtotn_predict':qtotn_predict_denorm.data.numpy(),
            'qtotn_test':qtotn_test_denorm.data.numpy(), 
            # 'thetan_predict':thetan_predict_denorm.data.numpy(),
            # 'thetan_test':thetan_test_denorm.data.numpy(),
            'qtotn_predict_norm':qtotn_predict_norm.data.numpy(),
            'qtotn_test_norm':qtotn_test_norm.data.numpy(), 
            # 'thetan_predict_norm':thetan_predict_norm.data.numpy(),
            # 'thetan_test_norm':thetan_test_norm.data.numpy(),
            # 'qphys_ml':qphys_ml.data.numpy(), 
            # 'qphys':qphys_denorm.data.numpy(),
            # 'theta_phys_ml':theta_phys_ml.data.numpy(), 
            # 'theta_phys':theta_phys_denorm.data.numpy(),
            'qtot':qtot_denorm.data.numpy(), 
            'theta':theta_denorm.data.numpy()
            }
    
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def evaluate_tnext(model, datasetfile, args):
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'], 
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False)
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    yp = model(x)
    xinv = nn_data._inverse_transform(x,xmean,xstd)
    xinv_split = nn_data.split_data(xinv,xyz='x')
    yinv = nn_data._inverse_transform(y,ymean,ystd)

    ypinv = nn_data._inverse_transform(yp,ymean,ystd)
    yinv_split = nn_data.split_data(yinv,xyz='y')
    ypinv_split = nn_data.split_data(ypinv, xyz='y')
    y_split = nn_data.split_data(y, xyz='y')
    yp_split = nn_data.split_data(yp, xyz='y')

    y2inv = nn_data._inverse_transform(y2,ymean2,ystd2)
    y2inv_split = nn_data.split_data(y2inv, xyz='y2')
    # ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
    # ['qphys', 'theta_phys']
    qtot_denorm = xinv_split['qtot']
    theta_denorm = xinv_split['theta']
    # qadv_denorm = xinv_split['qadv']
    # theta_adv_denorm = xinv_split['theta_adv']

    qtotn_predict_denorm = ypinv_split['qtot_next']
    qtotn_test_denorm = yinv_split['qtot_next']
    thetan_predict_denorm = ypinv_split['theta_next']
    thetan_test_denorm = yinv_split['theta_next']
    qtotn_predict_norm = yp_split['qtot_next']
    qtotn_test_norm = y_split['qtot_next']
    thetan_predict_norm = yp_split['theta_next']
    thetan_test_norm = y_split['theta_next']
    
    qphys_denorm = y2inv_split['qphys']
    theta_phys_denorm = y2inv_split['theta_phys']
    # qphys_ml = qtotn_predict_denorm - qtot_denorm - qadv_denorm
    # theta_phys_ml = thetan_predict_denorm - theta_denorm - theta_adv_denorm

   
    hfilename = args.model_name.replace('.tar','_qtphys.hdf5')

    output={'qtotn_predict':qtotn_predict_denorm.data.numpy(),
            'qtotn_test':qtotn_test_denorm.data.numpy(), 
            'thetan_predict':thetan_predict_denorm.data.numpy(),
            'thetan_test':thetan_test_denorm.data.numpy(),
            'qtotn_predict_norm':qtotn_predict_norm.data.numpy(),
            'qtotn_test_norm':qtotn_test_norm.data.numpy(), 
            'thetan_predict_norm':thetan_predict_norm.data.numpy(),
            'thetan_test_norm':thetan_test_norm.data.numpy(),
            # 'qphys_ml':qphys_ml.data.numpy(), 
            'qphys':qphys_denorm.data.numpy(),
            # 'theta_phys_ml':theta_phys_ml.data.numpy(), 
            'theta_phys':theta_phys_denorm.data.numpy(),
            'qtot':qtot_denorm.data.numpy(), 
            'theta':theta_denorm.data.numpy()
            }
    
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = model_loc+"qnext_004_in_045_out_010_epch_00500_btch_023001AQT_mae_163001AQT_normalise_cnn.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_000.hdf5"
    normaliser_region = "163001AQT_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    # evaluate_qtphys(model, datasetfile, args)
    # evaluate_add_adv(model, datasetfile, args)
    evaluate_qnext(model, datasetfile, args)
    # evaluate_qt_next(model, datasetfile, args)