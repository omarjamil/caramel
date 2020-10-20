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
import matplotlib.pyplot as plt

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

def set_model_q(model_file, args):

    in_features = args.in_features
    print(in_features)
    nb_classes = args.nb_classes
    nb_hidden_layers = args.nb_hidden_layers
    hidden_size = args.hidden_size
    # mlp = nn_model.MLP(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_tanh(in_features, nb_classes, nb_hidden_layers, hidden_size, scale=1.)
    mlp = nn_model.MLP_RELU(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.ResMLP(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = nn_model.MLP_sig(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_BN_tanh(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_multiout_tanh(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # skipindx = list(range(args.nlevs))
    # mlp = nn_model.MLPSubSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size, skipindx)
    # mlp = nn_model.MLPSkip(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLPSkip_(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLPDrop(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_BN(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # Load the save model 
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval() 
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    return mlp

def set_model_t(model_file, args):

    in_features = args.in_features
    print(in_features)
    nb_classes = args.nb_classes
    nb_hidden_layers = args.nb_hidden_layers
    hidden_size = args.hidden_size
    # mlp = nn_model.MLP(in_features, nb_classes, nb_hidden_layers, hidden_size)
    mlp = nn_model.MLP_tanh(in_features, nb_classes, nb_hidden_layers, hidden_size, scale=1.)
    # mlp = nn_model.MLP_sig(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_BN_tanh(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_multiout_tanh(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # skipindx = list(range(args.nlevs))
    # mlp = nn_model.MLPSubSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size, skipindx)
    # mlp = nn_model.MLPSkip(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLPSkip_(in_features, nb_classes, nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLPDrop(in_features, nb_classes, args.nb_hidden_layers, hidden_size)
    # mlp = nn_model.MLP_BN(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # Load the save model 
    print("Loading PyTorch model: {0}".format(model_file))
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval() 
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    return mlp

def plotX(xcopy, args):
    fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True, sharey=True)
    ax = axs[0]
    c = ax.pcolor(xcopy.data.numpy()[:,:args.nlevs].T)
    ax.set_title('q')
    fig.colorbar(c,ax=ax)

    # ax = axs[1]
    # c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs:args.nlevs*2].T)
    # ax.set_title('theta')
    # fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs:args.nlevs*2].T)
    ax.set_title('P')
    fig.colorbar(c,ax=ax)

    ax = axs[2]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*2:args.nlevs*3].T)
    ax.set_title('rho')
    fig.colorbar(c,ax=ax)

    fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True, sharey=True)
    ax = axs[0]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*3:args.nlevs*4].T)
    ax.set_title('xwind')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*4:args.nlevs*5].T)
    ax.set_title('ywind')
    fig.colorbar(c,ax=ax)

    ax = axs[2]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*5:args.nlevs*6].T)
    ax.set_title('zwind')
    fig.colorbar(c,ax=ax)

    fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True)
    ax = axs[0]
    c = ax.plot(xcopy.data.numpy()[:,args.nlevs*6:(args.nlevs*6)+1][:,0])
    ax.set_title('shf')

    ax = axs[1]
    
    c = ax.plot(xcopy.data.numpy()[:,(args.nlevs*6)+1:(args.nlevs*6)+2][:,0])
    ax.set_title('lhf')

    ax = axs[2]
    c = ax.plot(xcopy.data.numpy()[:,(args.nlevs*6)+2:(args.nlevs*6)+3][:,0])
    ax.set_title('swtoa')
    
    plt.show()    

def scm_diff_adv(qmodel, tmodel, datasetfile, qargs, targs):
    """
    SCM type run with two models
    one model for q prediction and nother for theta perdiction
    """
    qnn_data = data_io.Data_IO_validation_x2(qargs.region, qargs.nlevs, datasetfile, qargs.locations['normaliser_loc'],
                        xvars=qargs.xvars,
                        yvars=qargs.yvars,
                        xvars2=qargs.xvars2,
                        no_norm = True)

    tnn_data = data_io.Data_IO_validation_x2(targs.region, targs.nlevs, datasetfile, targs.locations['normaliser_loc'],
                        xvars=targs.xvars,
                        yvars=targs.yvars,
                        xvars2=targs.xvars2,
                        no_norm = True)                  
    print("Q model x: ", qargs.xvars)
    print("Q model xmul", qargs.xvar_multiplier)
    print("Q model x: ", qargs.xvars2)
    print("Q Model y: ", qargs.yvars)
    print("Q Model ymul: ", qargs.yvar_multiplier)
    print("T Model x: ", targs.xvars)
    print("T Model xmul: ", targs.xvar_multiplier)
    print("T model x: ", targs.xvars2)
    print("T Model y: ", targs.yvars)
    print("T Model y: ", targs.yvar_multiplier)
    qx,qy,qx2,qxmean,qxstd,qymean,qystd,qymean2,qystd2 = qnn_data.get_data()
    tx,ty,tx2,txmean,txstd,tymean,tystd,tymean2,tystd2 = tnn_data.get_data()
    # model = set_model(args)
    qx_split = qnn_data.split_data(qx,xyz='x')
    qyt_split = qnn_data.split_data(qy, xyz='y')
    tx_split = tnn_data.split_data(tx,xyz='x')
    tyt_split = tnn_data.split_data(ty, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    qyt_inverse = qy
    qyt_inverse_split = qnn_data.split_data(qyt_inverse, xyz='y')
    tyt_inverse = ty
    tyt_inverse_split = tnn_data.split_data(tyt_inverse, xyz='y')
    qnext = qyt_split['qtot'][:-1]
    qnext_inv = qyt_inverse_split['qtot'][:-1]
    tnext = tyt_split['theta'][:-1]
    tnext_inv = tyt_inverse_split['theta'][:-1]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    qx_inv_split = qnn_data.split_data(qx,xyz='x')
    tx_inv_split = tnn_data.split_data(tx,xyz='x')
    qtot_inv = tx_inv_split['qtot'][:-1]
    theta_inv = qx_inv_split['theta'][:-1]
    
    qoutput = []
    qoutput.append(qnext[0])
    # qoutput.append(qnext[1])

    toutput = []
    toutput.append(tnext[0])
    # toutput.append(tnext[1])

    qdifflist = []
    for v,m in zip(qargs.xvars, qargs.xvar_multiplier):
        qvdiff = (qx_split[v][1:] - qx_split[v][:-1])*m
        qdifflist.append(qvdiff[:-1])
    # advected quantity
    qdifflist.append(qx2[1:-1])

    tdifflist = []
    for v,m in zip(targs.xvars, targs.xvar_multiplier):
        tvdiff = (tx_split[v][1:] - tx_split[v][:-1])*m
        tdifflist.append(tvdiff[:-1])
    tdifflist.append(tx2[1:-1])

    qxcopy = torch.cat(qdifflist, dim=1)
    txcopy = torch.cat(tdifflist, dim=1)
    qdiff_output = []
    qdiff = tdifflist[0]
    qdiff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = qdifflist[0]
    tdiff_output.append((tdiff[0]))
    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qdiff)-1,1):
        
        qyp_ = qmodel(qxcopy[t-1])
        typ_ = tmodel(txcopy[t-1])
        # qyp_ = (qyp_ + qdiff_output[t-1])/2.
        # typ_ = (typ_ + tdiff_output[t-1])/2.
        qnew = qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
        tnew = toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]

        negative_values = torch.where(qnew < 0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]

        qoutput.append(qnew)
        qdiff_output.append(qyp_)
        qxcopy[t,:qargs.nlevs] = typ_*10.
        # qxcopy[t,:qargs.nlevs] = qyp_
        # qxcopy[t,qargs.nlevs:qargs.nlevs*2] = typ_
        toutput.append(tnew)
        tdiff_output.append(typ_[:targs.nlevs])
        txcopy[t,:targs.nlevs] = qyp_*10.
        # txcopy[t,:targs.nlevs] = typ_*10.


    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qyp = torch.stack(qoutput)
    typ = torch.stack(toutput)
  
    qnext_ml_inv = qyp
    tnext_ml_inv = typ

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = targs.model_name.replace('.tar','_scm_2m.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)


def scm_diff_multiout(qmodel, tmodel, datasetfile, qargs, targs):
    """
    SCM type run with two models
    one model for q prediction and nother for theta perdiction
    """
    qnn_data = data_io.Data_IO_validation(qargs.region, qargs.nlevs, datasetfile, qargs.locations['normaliser_loc'],
                        xvars=qargs.xvars,
                        yvars=qargs.yvars,
                        # yvars2=qargs.yvars2,
                        add_adv=False, 
                        no_norm = False)

    tnn_data = data_io.Data_IO_validation(targs.region, targs.nlevs, datasetfile, targs.locations['normaliser_loc'],
                        xvars=targs.xvars,
                        yvars=targs.yvars,
                        # yvars2=targs.yvars2,
                        add_adv=False, 
                        no_norm = False)           

    print("Q model x: ", qargs.xvars)
    print("Q model xmul", qargs.xvar_multiplier)
    print("Q Model y: ", qargs.yvars)
    print("Q Model ymul: ", qargs.yvar_multiplier)
    print("T Model x: ", targs.xvars)
    print("T Model xmul: ", targs.xvar_multiplier)
    print("T Model y: ", targs.yvars)
    print("T Model y: ", targs.yvar_multiplier)
    qx,qy,qy2,qxmean,qxstd,qymean,qystd,qymean2,qystd2 = qnn_data.get_data()
    tx,ty,ty2,txmean,txstd,tymean,tystd,tymean2,tystd2 = tnn_data.get_data()
    # model = set_model(args)
    qx_split = qnn_data.split_data(qx,xyz='x')
    qyt_split = qnn_data.split_data(qy, xyz='y')
    tx_split = tnn_data.split_data(tx,xyz='x')
    tyt_split = tnn_data.split_data(ty, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    qyt_inverse = qy
    # qyt_inverse = qnn_data._inverse_transform(qy,qymean,qystd)
    qyt_inverse_split = qnn_data.split_data(qyt_inverse, xyz='y')
    tyt_inverse = ty
    # tyt_inverse = tnn_data._inverse_transform(ty,tymean,tystd)
    tyt_inverse_split = tnn_data.split_data(tyt_inverse, xyz='y')
    qnext = qyt_split['qtot'][:-1]
    qnext_inv = qyt_inverse_split['qtot'][:-1]
    tnext = tyt_split['theta'][:-1]
    tnext_inv = tyt_inverse_split['theta'][:-1]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    qx_inv_split = qnn_data.split_data(qx,xyz='x')
    tx_inv_split = tnn_data.split_data(tx,xyz='x')
    qtot_inv = tx_inv_split['qtot'][:-1]
    theta_inv = qx_inv_split['theta'][:-1]
    
    qoutput = []
    qoutput.append(qnext[0])
    # qoutput.append(qnext[1])

    toutput = []
    toutput.append(tnext[0])
    # toutput.append(tnext[1])

    qdifflist = []
    for v,m in zip(qargs.xvars, qargs.xvar_multiplier):
        qvdiff = (qx_split[v][1:] - qx_split[v][:-1])*m
        qdifflist.append(qvdiff[:-1])

    tdifflist = []
    for v,m in zip(targs.xvars, targs.xvar_multiplier):
        tvdiff = (tx_split[v][1:] - tx_split[v][:-1])*m
        tdifflist.append(tvdiff[:-1])

    qxcopy = torch.cat(qdifflist, dim=1)
    txcopy = torch.cat(tdifflist, dim=1)
    qdiff_output = []
    qdiff = tdifflist[0]
    qdiff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = qdifflist[0]
    tdiff_output.append((tdiff[0]))
    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qdiff)-1,1):
        
        # qyp_ = qmodel(qxcopy[t-1])
        # qyp_1, qyp_2, qyp_3, qyp_4 = qmodel(qxcopy[t-1])
        # qyp_ = qyp_1*0.50 + qyp_2*0.50 + qyp_3*0.0 + qyp_4*0.0
        typ_1, typ_2, typ_3 = tmodel(txcopy[t-1])
        typ_ = (typ_1 + typ_2 + typ_3)/3.
        # typ_ = typ_2
        qxcopy[t-1,:qargs.nlevs] = typ_*10.

        qyp_1, qyp_2, qyp_3 = qmodel(qxcopy[t-1])
        qyp_ = (qyp_1 + qyp_2 + qyp_3)/3.
        # qyp_ = qyp_2
        txcopy[t,:targs.nlevs] = qyp_*10.

        # qyp_ = qyp_1*0.33 + qyp_2*0.33 + qyp_3*0.33
        # typ_ = tmodel(txcopy[t-1])
        # qyp_ = (qyp_ + qdiff_output[t-1])/2.
        # typ_ = (typ_ + tdiff_output[t-1])/2.
        # qnew = qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
        qnew1 = qoutput[t-1] + qyp_1[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
        qnew2 = qoutput[t-1] + qyp_2[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
        qnew3 = qoutput[t-1] + qyp_3[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
        qnew = (qnew1+qnew2+qnew3)/3.
        # qnew = (qnew2+qnew3)/2.
        # qnew = qnew2
        # qnew = (qnew2 + qnew3)/2.
        # qyp_ = (qoutput[t-1] - qnew)*10.
        # qyp_ = qyp_2
        # qyp_ = (qyp_1 + qyp_2 + qyp_3)/3.
        # print(t)
        # print("Q", qyp_[0], qyp_1[0], qyp_2[0], qyp_3[0], txcopy[t,0])
        # qyp_ = (qyp_1 + qyp_2 + qyp_3)/3.
        # tnew = toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]
        tnew1 = toutput[t-1] + typ_1[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]
        tnew2 = toutput[t-1] + typ_2[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]
        tnew3 = toutput[t-1] + typ_3[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]
        # tnew = (tnew2 + tnew3)/2.
        tnew = (tnew1 + tnew2 + tnew3)/3.
        # tnew = tnew2
        # typ_ = (toutput[t-1] - tnew)*10.
        # typ_ = (typ_1 + typ_2 + typ_3)/3.
        # typ_ = typ_2
        # print("T", typ_[0], typ_1[0], typ_2[0], typ_3[0], qxcopy[t,0])

        negative_values = torch.where(qnew < 0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]

        qoutput.append(qnew)
        qdiff_output.append(qyp_)
        # qxcopy[t,:qargs.nlevs] = typ_*10.
        # qxcopy[t,:qargs.nlevs] = qyp_
        # qxcopy[t,qargs.nlevs:qargs.nlevs*2] = typ_
        toutput.append(tnew)
        tdiff_output.append(typ_[:targs.nlevs])
        # txcopy[t,:targs.nlevs] = typ_*10.


    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qyp = torch.stack(qoutput)
    typ = torch.stack(toutput)
  
    qnext_ml_inv = qyp
    tnext_ml_inv = typ

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = qargs.model_name.replace('.tar','_scm_2m.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def scm_diff_normed(qmodel, tmodel, datasetfile, qargs, targs, subdomain):
    """
    SCM type run with two models
    one model for q prediction and nother for theta perdiction
    """
    q_initial = torch.Tensor([1.63182076e-02, 1.62280705e-02, 1.60901472e-02, 1.58688519e-02,
                            1.55681688e-02, 1.52368555e-02, 1.48642398e-02, 1.44164069e-02,
                            1.38758617e-02, 1.35183409e-02, 1.31579693e-02, 1.23914201e-02,
                            1.12009374e-02, 1.04784751e-02, 1.01551162e-02, 9.45019070e-03,
                            8.36965442e-03, 6.58852793e-03, 5.01506496e-03, 4.64849453e-03,
                            4.57216706e-03, 4.13405523e-03, 3.70655162e-03, 3.03756632e-03,
                            1.43692654e-03, 5.42424386e-04, 4.48503095e-04, 3.97078198e-04,
                            4.67831967e-04, 3.40979226e-04, 2.68895557e-04, 3.50894465e-04,
                            3.17292026e-04, 1.71423890e-04, 1.06021958e-04, 1.00951263e-04,
                            1.05092397e-04, 1.07926127e-04, 1.09760760e-04, 1.12764967e-04,
                            6.90982051e-05, 1.91042163e-05, 9.23567313e-06, 7.77933292e-06,
                            4.80280960e-06, 3.51210429e-06, 2.65399626e-06, 2.50743506e-06,
                            2.50339508e-06, 2.50988546e-06, 2.89142122e-06, 3.69674626e-06,
                            4.34709909e-06, 4.47034836e-06, 4.35113907e-06, 4.05311584e-06,
                            3.93390656e-06, 3.93390656e-06, 3.87430191e-06, 3.87430191e-06,
                            3.87430191e-06, 3.87430191e-06, 3.75509262e-06, 3.63588333e-06,
                            3.42216754e-06, 2.98023224e-06, 2.56386079e-06, 2.50339508e-06,
                            2.50339508e-06, 2.50339508e-06])
    theta_initial = torch.Tensor([295.1643 ,  295.26263,  295.39307,  295.59167,  295.82916,
                                296.11987,  296.5157 ,  297.0018 ,  297.54932,  298.00513,
                                298.64847,  299.9075 ,  302.72916,  303.64584,  304.8786 ,
                                307.1372 ,  309.22098,  311.27902,  313.19138,  314.6157 ,
                                315.3718 ,  315.82904,  316.9307 ,  318.78055,  321.48682,
                                324.10306,  325.93985,  327.61612,  329.84113,  332.03265,
                                334.4818 ,  335.70737,  336.4207 ,  337.34402,  338.83765,
                                339.97015,  340.18152,  340.15875,  340.0982 ,  339.85056,
                                343.28235,  351.57584,  354.10568,  356.1489 ,  358.06876,
                                359.50903,  362.64542,  368.47446,  376.79904,  389.80292,
                                427.89777,  463.50235,  503.3    ,  541.9146 ,  572.918  ,
                                611.853  ,  662.0325 ,  732.4021 ,  819.6621 ,  916.9728 ,
                                1041.1086 , 1203.8889 , 1427.1561 , 1702.4939 , 2000.9613 ,
                                2378.9746 , 2843.456  , 3395.4312 , 3985.388  , 5040.898 ])

    qnn_data = data_io.Data_IO_validation_diff(qargs.region, qargs.nlevs, datasetfile, qargs.locations['normaliser_loc'],
                        xvars=qargs.xvars,
                        yvars=qargs.yvars,
                        # yvars2=qargs.yvars2,
                        add_adv=False, 
                        no_norm = False)

    tnn_data = data_io.Data_IO_validation_diff(targs.region, targs.nlevs, datasetfile, targs.locations['normaliser_loc'],
                        xvars=targs.xvars,
                        yvars=targs.yvars,
                        # yvars2=targs.yvars2,
                        add_adv=False, 
                        no_norm = False)

    print("Q model x: ", qargs.xvars)
    print("Q model xmul", qargs.xvar_multiplier)
    print("Q Model y: ", qargs.yvars)
    print("Q Model ymul: ", qargs.yvar_multiplier)
    print("T Model x: ", targs.xvars)
    print("T Model xmul: ", targs.xvar_multiplier)
    print("T Model y: ", targs.yvars)
    print("T Model y: ", targs.yvar_multiplier)
    qx,qy,qy2,qxmean,qxstd,qymean,qystd,qymean2,qystd2 = qnn_data.get_data()
    tx,ty,ty2,txmean,txstd,tymean,tystd,tymean2,tystd2 = tnn_data.get_data()
    # model = set_model(args)
    qx_split = qnn_data.split_data(qx,xyz='x')
    qyt_split = qnn_data.split_data(qy, xyz='y')
    tx_split = tnn_data.split_data(tx,xyz='x')
    tyt_split = tnn_data.split_data(ty, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    # qyt_inverse = qy
    qyt_inverse = qnn_data._inverse_transform(qy,qymean,qystd)
    qyt_inverse_split = qnn_data.split_data(qyt_inverse, xyz='y')
    # tyt_inverse = ty
    tyt_inverse = tnn_data._inverse_transform(ty,tymean,tystd)
    tyt_inverse_split = tnn_data.split_data(tyt_inverse, xyz='y')
   
    qxcopy = qx
    txcopy = tx
    qdiff_output = []
    qdiff = tx_split['qtot']
    qdiff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = qx_split['theta']
    tdiff_output.append((tdiff[0]))
    # print(qx[:10,:10])
    # plotX(qxcopy, qargs)
    # sys.exit(0)

    for t in range(1,len(qdiff)-1,1):
        print(t)
        typ_ = tmodel(txcopy[t-1].reshape(1,-1))
        # print("ML",typ_[0,:])
        # print("T", ty[t,:])
        # if t > 2:
            # qxcopy[t-1,:qargs.nlevs] = typ_
        qyp_= qmodel(qxcopy[t-1].reshape(1,-1))
        qyp_inv = qnn_data._inverse_transform(qyp_,qymean,qystd)
        qy_inv = qnn_data._inverse_transform(qy,qymean,qystd)
        print("qML inv",qyp_inv[0,:])
        print("Q inv", qy_inv[t,:])
        print("diffq inv",qyp_inv[0,:]-qy_inv[t,:])
        print("qML",qyp_[0,:])
        print("Q", qy[t,:])
        print("diffq",qyp_[0,:]-qy[t,:])
        qdiff_output.append(qyp_.reshape(-1))
        tdiff_output.append(typ_.reshape(-1))
        # if t > 2:
            # txcopy[t,:targs.nlevs] = qyp_

    qoutput = []
    qoutput.append(q_initial[:qargs.nlevs])
    toutput = []
    toutput.append(theta_initial[:targs.nlevs])
    qdiff_output = torch.stack(qdiff_output)
    tdiff_output = torch.stack(tdiff_output)
    # print(qdiff_output[:10,0])
    # print(qy[:10,0])
    # print(qyt_inverse[:10,0])
    qdiff_pred = qnn_data._inverse_transform(qdiff_output,qymean,qystd)
   
    tdiff_pred = tnn_data._inverse_transform(tdiff_output,tymean,tystd)
   
    q_next = []
    t_next = []
    q_next.append(q_initial[:qargs.nlevs])
    t_next.append(theta_initial[:targs.nlevs])

    for t in range(1,len(qdiff_output)):
        if t <= 2:
            qml = qoutput[t-1] + qyt_inverse[t]
            tml = toutput[t-1] + tyt_inverse[t]
        else:
            qml = qoutput[t-1] + qdiff_pred[t]
            tml = toutput[t-1] + tdiff_pred[t]
        qoutput.append(qml)
        toutput.append(tml)

        qtrue = q_next[t-1] + qyt_inverse[t]
        ttrue = t_next[t-1] + tyt_inverse[t]
        q_next.append(qtrue)
        t_next.append(ttrue)
    qyp = torch.stack(qoutput)
    typ = torch.stack(toutput)
    qtrue = torch.stack(q_next)
    ttrue = torch.stack(t_next)
    
    
    output = {
            'qtot_next':qtrue.data.numpy(), 
            'qtot_next_ml':qyp.data.numpy(),
            'qtot':qtrue.data.numpy(),
            'theta':ttrue.data.numpy(),
            'theta_next':ttrue.data.numpy(), 
            'theta_next_ml':typ.data.numpy()
    }
    hfilename = qargs.model_name.replace('.tar','_scm_2m_{0}.hdf5'.format(str(subdomain).zfill(3))) 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def scm_diff_enc(qmodel, tmodel, encmodelfile, datasetfile, qargs, targs, subdomain):
    """
    SCM type run with two models
    one model for q prediction and nother for theta perdiction
    """

    aemodel = nn_model.AE(qargs.nlevs)
    print(aemodel)
    print("Loading PyTorch model: {0}".format(encmodelfile))
    checkpoint = torch.load(encmodelfile, map_location=torch.device('cpu'))
    for param_tensor in checkpoint['model_state_dict']:
        print(param_tensor, "\t", checkpoint['model_state_dict'][param_tensor].size())
    # print(checkpoint['model_state_dict'])
    aemodel.load_state_dict(checkpoint['model_state_dict'])
    aemodel.eval() 
    print("Model's state_dict:")
    for param_tensor in aemodel.state_dict():
        print(param_tensor, "\t", aemodel.state_dict()[param_tensor].size())

    qnn_data = data_io.Data_IO_validation(qargs.region, qargs.nlevs, datasetfile, qargs.locations['normaliser_loc'],
                        xvars=qargs.xvars,
                        yvars=qargs.yvars,
                        # yvars2=qargs.yvars2,
                        add_adv=False, 
                        no_norm = False,
                        fmin=0,
                        fmax=100)

    tnn_data = data_io.Data_IO_validation(targs.region, targs.nlevs, datasetfile, targs.locations['normaliser_loc'],
                        xvars=targs.xvars,
                        yvars=targs.yvars,
                        # yvars2=targs.yvars2,
                        add_adv=False, 
                        no_norm = False,
                        fmin=0,
                        fmax=100)

    # mplier = targs.xvar_multiplier
    # mplier[0] = 4.
    # targs.xvar_multiplier = mplier
    print("Q model x: ", qargs.xvars)
    print("Q model xmul", qargs.xvar_multiplier)
    print("Q Model y: ", qargs.yvars)
    print("Q Model ymul: ", qargs.yvar_multiplier)
    print("T Model x: ", targs.xvars)
    print("T Model xmul: ", targs.xvar_multiplier)
    print("T Model y: ", targs.yvars)
    print("T Model y: ", targs.yvar_multiplier)
    qx,qy,qy2,qxmean,qxstd,qymean,qystd,qymean2,qystd2 = qnn_data.get_data()
    tx,ty,ty2,txmean,txstd,tymean,tystd,tymean2,tystd2 = tnn_data.get_data()
    # model = set_model(args)
    qx_split = qnn_data.split_data(qx,xyz='x')
    qyt_split = qnn_data.split_data(qy, xyz='y')
    tx_split = tnn_data.split_data(tx,xyz='x')
    tyt_split = tnn_data.split_data(ty, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    # qyt_inverse = qy
    qyt_inverse = qnn_data._inverse_transform(qy,qymean,qystd)
    qyt_inverse_split = qnn_data.split_data(qyt_inverse, xyz='y')
    # tyt_inverse = ty
    tyt_inverse = tnn_data._inverse_transform(ty,tymean,tystd)
    tyt_inverse_split = tnn_data.split_data(tyt_inverse, xyz='y')
    qnext = qyt_split['qtot'][:-1]
    qnext_inv = qyt_inverse_split['qtot'][:-1]
    tnext = tyt_split['theta'][:-1]
    tnext_inv = tyt_inverse_split['theta'][:-1]

    tx_inv_split = tnn_data._inverse_transform_split(tx_split,txmean,txstd,xyz='x')
    qx_inv_split = qnn_data._inverse_transform_split(qx_split,qxmean,qxstd,xyz='x')
    # qx_inv_split = qnn_data.split_data(qx,xyz='x')
    # tx_inv_split = tnn_data.split_data(tx,xyz='x')
    qtot_inv = tx_inv_split['qtot'][:-1]
    theta_inv = qx_inv_split['theta'][:-1]
    
    qoutput = []
    qoutput.append(qnext[0])
    # qoutput.append(qnext[1])

    toutput = []
    toutput.append(tnext[0])
    # toutput.append(tnext[1])

    qdifflist = []
    
    for v,m in zip(qargs.xvars, qargs.xvar_multiplier):
        qvdiff = (qx_split[v][1:] - qx_split[v][:-1])*m
        qdifflist.append(qvdiff[:-1])

    tdifflist = []
   
    for v,m in zip(targs.xvars, targs.xvar_multiplier):
        tvdiff = (tx_split[v][1:] - tx_split[v][:-1])*m
        tdifflist.append(tvdiff[:-1])

    qxcopy = torch.cat(qdifflist, dim=1)
    txcopy = torch.cat(tdifflist, dim=1)
    qdiff_output = []
    qdiff = tdifflist[0]
    qdiff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = qdifflist[0]
    tdiff_output.append((tdiff[0]))
    # plotX(txcopy, qargs)
    # sys.exit(0)

    for t in range(1,len(qdiff)-1,1):

        typ_ = tmodel(txcopy[t-1].reshape(1,-1))
        if t > 2:
            qxcopy[t-1,:qargs.nlevs] = typ_
            # qxcopy[t-1,:qargs.nlevs] = qdiff_output[t-2]
            # qxcopy[t-1,qargs.nlevs:qargs.nlevs*2] = typ_

        # true_enc = aemodel.encoder(txcopy[t-1][:qargs.nlevs])
        qyp_enc = qmodel(qxcopy[t-1].reshape(1,-1))
        # print("True", true_enc[:])
        # print("Model", qyp_enc[:])
        qyp_ = aemodel.decoder(qyp_enc)
        qxcopy[t,:qargs.nlevs] = qyp_
        if t > 2:
            txcopy[t,:targs.nlevs] = qyp_

        if t > 2:
            qnew = (qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:])*1.0+(qoutput[t-2]*0.0)
            tnew = (toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:])*1.0+(toutput[t-2]*0.0)

        else:
            qnew = qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
            tnew = toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]

        negative_values = torch.where(qnew < 0.)
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]

        # qoutput.append(qnew)

        qoutput.append(qnew.reshape(-1))
        qdiff_output.append(qyp_)
        toutput.append(tnew.reshape(-1))
        tdiff_output.append(typ_[:targs.nlevs])

    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qyp = torch.stack(qoutput)
    typ = torch.stack(toutput)
  
    qnext_ml_inv = qnn_data._inverse_transform(qyp,qymean,qystd)
    tnext_ml_inv = tnn_data._inverse_transform(typ,tymean,tystd)
    
    
    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = qargs.model_name.replace('.tar','_scm_2m_{0}.hdf5'.format(str(subdomain).zfill(3))) 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def scm_diff(qmodel, tmodel, datasetfile, qargs, targs, subdomain):
    """
    SCM type run with two models
    one model for q prediction and nother for theta perdiction
    """
    qnn_data = data_io.Data_IO_validation(qargs.region, qargs.nlevs, datasetfile, qargs.locations['normaliser_loc'],
                        xvars=qargs.xvars,
                        yvars=qargs.yvars,
                        # yvars2=qargs.yvars2,
                        add_adv=False, 
                        no_norm = False,
                        fmin=0,
                        fmax=100)

    tnn_data = data_io.Data_IO_validation(targs.region, targs.nlevs, datasetfile, targs.locations['normaliser_loc'],
                        xvars=targs.xvars,
                        yvars=targs.yvars,
                        # yvars2=targs.yvars2,
                        add_adv=False, 
                        no_norm = False,
                        fmin=0,
                        fmax=100)

    # mplier = targs.xvar_multiplier
    # mplier[0] = 4.
    # targs.xvar_multiplier = mplier
    print("Q model x: ", qargs.xvars)
    print("Q model xmul", qargs.xvar_multiplier)
    print("Q Model y: ", qargs.yvars)
    print("Q Model ymul: ", qargs.yvar_multiplier)
    print("T Model x: ", targs.xvars)
    print("T Model xmul: ", targs.xvar_multiplier)
    print("T Model y: ", targs.yvars)
    print("T Model y: ", targs.yvar_multiplier)
    qx,qy,qy2,qxmean,qxstd,qymean,qystd,qymean2,qystd2 = qnn_data.get_data()
    tx,ty,ty2,txmean,txstd,tymean,tystd,tymean2,tystd2 = tnn_data.get_data()
    # model = set_model(args)
    qx_split = qnn_data.split_data(qx,xyz='x')
    qyt_split = qnn_data.split_data(qy, xyz='y')
    tx_split = tnn_data.split_data(tx,xyz='x')
    tyt_split = tnn_data.split_data(ty, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    # qyt_inverse = qy
    qyt_inverse = qnn_data._inverse_transform(qy,qymean,qystd)
    qyt_inverse_split = qnn_data.split_data(qyt_inverse, xyz='y')
    # tyt_inverse = ty
    tyt_inverse = tnn_data._inverse_transform(ty,tymean,tystd)
    tyt_inverse_split = tnn_data.split_data(tyt_inverse, xyz='y')
    qnext = qyt_split['qtot'][:-1]
    qnext_inv = qyt_inverse_split['qtot'][:-1]
    tnext = tyt_split['theta'][:-1]
    tnext_inv = tyt_inverse_split['theta'][:-1]

    tx_inv_split = tnn_data._inverse_transform_split(tx_split,txmean,txstd,xyz='x')
    qx_inv_split = qnn_data._inverse_transform_split(qx_split,qxmean,qxstd,xyz='x')
    # qx_inv_split = qnn_data.split_data(qx,xyz='x')
    # tx_inv_split = tnn_data.split_data(tx,xyz='x')
    qtot_inv = tx_inv_split['qtot'][:-1]
    theta_inv = qx_inv_split['theta'][:-1]
    
    qoutput = []
    qoutput.append(qnext[0])
    # qoutput.append(qnext[1])

    toutput = []
    toutput.append(tnext[0])
    # toutput.append(tnext[1])

    qdifflist = []
    
    for v,m in zip(qargs.xvars, qargs.xvar_multiplier):
        qvdiff = (qx_split[v][1:] - qx_split[v][:-1])*m
        qdifflist.append(qvdiff[:-1])

    tdifflist = []
   
    for v,m in zip(targs.xvars, targs.xvar_multiplier):
        tvdiff = (tx_split[v][1:] - tx_split[v][:-1])*m
        tdifflist.append(tvdiff[:-1])

    qxcopy = torch.cat(qdifflist, dim=1)
    txcopy = torch.cat(tdifflist, dim=1)
    qdiff_output = []
    qdiff = tdifflist[0]
    qdiff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = qdifflist[0]
    tdiff_output.append((tdiff[0]))
    # plotX(txcopy, qargs)
    # sys.exit(0)

    for t in range(1,len(qdiff)-1,1):

        typ_ = tmodel(txcopy[t-1].reshape(1,-1))
        if t > 2:
            qxcopy[t-1,:qargs.nlevs] = typ_
            # qxcopy[t-1,:qargs.nlevs] = qdiff_output[t-2]
            # qxcopy[t-1,qargs.nlevs:qargs.nlevs*2] = typ_

        qyp_= qmodel(qxcopy[t-1].reshape(1,-1))
        qxcopy[t,:qargs.nlevs] = qyp_
        if t > 2:
            txcopy[t,:targs.nlevs] = qyp_

        if t > 2:
            qnew = (qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:])*1.0+(qoutput[t-2]*0.0)
            tnew = (toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:])*1.0+(toutput[t-2]*0.0)

        else:
            qnew = qoutput[t-1] + qyp_[:qargs.nlevs]/torch.Tensor(qargs.yvar_multiplier)[:]
            tnew = toutput[t-1] + typ_[:targs.nlevs]/torch.Tensor(targs.yvar_multiplier)[:]

        negative_values = torch.where(qnew < 0.)
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]

        # qoutput.append(qnew)

        qoutput.append(qnew.reshape(-1))
        qdiff_output.append(qyp_)
        toutput.append(tnew.reshape(-1))
        tdiff_output.append(typ_[:targs.nlevs])

    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qyp = torch.stack(qoutput)
    typ = torch.stack(toutput)
  
    qnext_ml_inv = qnn_data._inverse_transform(qyp,qymean,qystd)
    tnext_ml_inv = tnn_data._inverse_transform(typ,tymean,tystd)
    
    
    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = qargs.model_name.replace('.tar','_scm_2m_{0}.hdf5'.format(str(subdomain).zfill(3))) 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)


def main(subdomain):
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    qmodel_file = model_loc+"qdiff_diag_normed_f0100_006_lyr_333_in_045_out_0378_hdn_050_epch_00150_btch_023001AQS_mse_sum_023001AQS_normalise_stkd_tstoch1sig_lr1e4_enc.tar"
    tmodel_file = model_loc+"tdiff_diag_normed_008_lyr_333_in_055_out_0388_hdn_050_epch_00150_btch_023001AQS_mse_sum_023001AQS_normalise_stkd_xstoch_tanh.tar"
    aemodel_file = model_loc+"qdiff_ae_stoch_normed_006_lyr_055_in_055_out_0110_hdn_050_epch_00150_btch_023001AQS_mse_023001AQS_normalise_stkd.tar"
    # datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100WD/validation_data_0N100WD_{0}.hdf5".format(str(subdomain).zfill(3))
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_{0}.hdf5".format(str(subdomain).zfill(3))
    # normaliser_region = "023001AQT_normalise_60_glb"
    # normaliser_region = "023001AQT_standardise_mx"
    # normaliser_region = "023001AQT_normalise"
    normaliser_region = "023001AQS_normalise"
    # normaliser_region = "023001AQSD_normalise"
    data_region = "0N100W"
    qargs = set_args(qmodel_file, normaliser_region, data_region)
    targs = set_args(tmodel_file, normaliser_region, data_region)
    qmodel = set_model_q(qmodel_file, qargs)
    tmodel = set_model_t(tmodel_file, targs)
    # scm_diff(qmodel, tmodel, datasetfile, qargs, targs, subdomain)
    scm_diff_enc(qmodel, tmodel, aemodel_file, datasetfile, qargs,targs,subdomain)
    # scm_diff_normed(qmodel, tmodel, datasetfile, qargs, targs, subdomain)
    # scm_diff_multiout(qmodel, tmodel, datasetfile, qargs, targs)
    # scm_diff_adv(qmodel, tmodel, datasetfile, qargs, targs)

if __name__ == "__main__":
    i = 63
    main(i)
    # for i in range(64):
        # main(i)
   