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
    print(args.xvars)
    args.region=args.data_region
    in_features = (args.nlevs*(len(args.xvars)-3)+3)
    if not args.train_on_y2:
        nb_classes = (args.nlevs*(len(args.yvars)))
    else:
        nb_classes = (args.nlevs*(len(args.yvars2)))
    # n_inputs,n_outputs=140,70
    # in_features, nb_classes=(args.nlevs*4+3),(args.nlevs*2)
    hidden_size = int(1. * in_features + nb_classes)
    # mlp = nn_model.MLP(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    mlp = nn_model.MLPSkip(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
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

def scm(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2, 
                        add_adv=False)
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    qnext_ml = x_split['qtot'].data.numpy()
    tnext_ml = x_split['theta'].data.numpy()
    q_in = x_split['qtot'][0]
    t_in = x_split['theta'][0]
    qadv, tadv, swtoa, shf, lhf = x_split['qadv'], x_split['theta_adv'], x_split['sw_toa'], x_split['shf'], x_split['lhf']
    # swtoa, shf, lhf = x_split['sw_toa'], x_split['shf'], x_split['lhf']

    for t in range(len(x)):
        # prediction
        # print("q_in, q_true: ", q_in.data.numpy()[0], x_split['qtot'].data.numpy()[t,0])
        qnext_ml[t] = q_in.data.numpy()[:]
        # tnext_ml[t] = t_in.data.numpy()[:]
        inputs = torch.cat([q_in,qadv[t],t_in,tadv[t],swtoa[t],shf[t],lhf[t]])
        # inputs = torch.cat([q_in,t_in,swtoa[t],shf[t],lhf[t]])
        yp_ = model(inputs)
        yp_split = nn_data.split_data(yp_,xyz='y')
        q_in = yp_split['qtot_next']
        # t_in = yp_split['theta_next']
        # print("q_ml:", q_in.data.numpy()[0])
    yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext_inv = yt_inverse_split['qtot_next']
    # tnext_inv = yt_inverse_split['theta_next']
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    yp = torch.from_numpy(qnext_ml)
    yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']


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
    model_file = model_loc+"qnext_006_lyr_183_in_045_out_0228_hdn_030_epch_00500_btch_023001AQT_mse_163001AQT_normalise_skip.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_063.hdf5"
    normaliser_region = "163001AQ_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    scm(model, datasetfile, args)