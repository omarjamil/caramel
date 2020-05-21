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

def set_args(model_file, normaliser_region, data_region):
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    args = checkpoint['arguments']
    args.isambard = False
    args.region = data_region
    args.device = torch.device('cpu')
    args.normaliser_region = normaliser_region
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


def scm_cnn(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
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
    
    qnext_ml = qtotn_test_norm.data.numpy().copy()
    # tnext_ml = x_split['theta'].data.numpy()
    q_in = y[0,:args.nlevs]
    t_in = x[:,2,:]
    qadv, tadv = x[:,1,:], x[:,3,:]
    swtoa, shf, lhf = x2[:,0], x2[:,1], x2[:,2]

    for t in range(len(x)):
        # prediction
        # print("q_in, q_true: ", q_in.data.numpy()[0], x_split['qtot'].data.numpy()[t,0])
        qnext_ml[t] = q_in.data.numpy()[:]
        # tnext_ml[t] = t_in.data.numpy()[:]
        inputs = torch.cat([q_in,qadv[t],t_in,tadv[t],swtoa[t],shf[t],lhf[t]])
        # inputs = torch.cat([q_in,t_in,swtoa[t],shf[t],lhf[t]])
        yp_ = model(inputs)
        q_in = yp_[:,:args.nlevs]
        # t_in = yp_[:,args.nlevs:]
        # print("q_ml:", q_in.data.numpy()[0])
    yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    qnext_inv = yt_inverse[:,:args.nlevs]
    # tnext_inv = yt_inverse[:,args.nlevs:]
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    yp = torch.from_numpy(qnext_ml)
    yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    qnext_ml_inv = yp_inverse[:,:args.nlevs]
    # tnext_ml_inv = yp_inverse[:,args.nlevs:]


    output = {'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            # 'theta_next':tnext_inv.data.numpy(), 
            # 'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = model_loc+"qnext_004_in_045_out_010_epch_00500_btch_023001AQT_mse_163001AQT_normalise_cnn.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_011.hdf5"
    normaliser_region = "163001AQ_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    # scm(model, datasetfile, args)
    scm_cnn(model, datasetfile, args)
    