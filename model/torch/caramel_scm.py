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

def set_model(model_file, args):

    # Define the Model
    # print(args.xvars)
    # args.region=args.data_region
    # in_features = (args.nlevs*(len(args.xvars)-3)+3)
    # if not args.train_on_y2:
    #     nb_classes = (args.nlevs*(len(args.yvars)))
    # else:
    #     nb_classes = (args.nlevs*(len(args.yvars2)))
    # # n_inputs,n_outputs=140,70
    # in_features, nb_classes=(args.nlevs*4+3),(args.nlevs*2)
    # hidden_size = int(1. * in_features + nb_classes)
    in_features = args.in_features
    print(in_features)
    nb_classes = args.nb_classes
    nb_hidden_layers = args.nb_hidden_layers
    hidden_size = args.hidden_size
    # mlp = nn_model.MLP(in_features, nb_classes, nb_hidden_layers, hidden_size)
    mlp = nn_model.MLP_tanh(in_features, nb_classes, nb_hidden_layers, hidden_size)
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
    # print("Running mean", mlp.state_dict()['bn_in.running_mean'])
    # print("Running var", mlp.state_dict()['bn_in.running_var'])
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
    recursion_indx = list(range(args.nlevs))
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext = yt_split['qtot_next']
    qnext_inv = yt_inverse_split['qtot_next']

    x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    qtot_inv = x_inv_split['qtot']
    
    output = []
    output.append(qnext[0])
    xcopy = x.clone()

    for t in range(1,len(x)-1):
        # next tstep prediction
        xcopy[t,recursion_indx] = output[t-1]
        output.append(model(xcopy[t]))
        print("Predict", output[t][...,0:10] - output[t-1][...,0:10])
        print("True", qnext[t,0:10] - qnext[t-1,0:10])
        
        # Difference prediction
        # xcopy[t,recursion_indx] = output[t-1]
        # yt = model(xcopy[t])/100.
        # output.append(yt + output[t-1])
        # print("Predict", yt[...,0:10])
        # print("True", qnext[t,0:10] - qnext[t-1,0:10])

    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    yp = torch.stack(output)
    # yp = torch.from_numpy(qnext_ml)
    yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    # yp_inverse = yp
    yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            # 'qtot':qtot.data.numpy(),
            # 'qtot_next':qnext.data.numpy(), 
            # 'qtot_next_ml':qnext_ml,
            # 'theta_next':tnext_inv.data.numpy(), 
            # 'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def plotX(xcopy):
    fig, axs = plt.subplots(4,1,figsize=(14, 10), sharex=True, sharey=True)
    ax = axs[0]
    c = ax.pcolor(xcopy.data.numpy()[:,:args.nlevs].T)
    ax.set_title('q')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs:args.nlevs*2].T)
    ax.set_title('theta')
    fig.colorbar(c,ax=ax)

    ax = axs[2]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*2:args.nlevs*3].T)
    ax.set_title('P')
    fig.colorbar(c,ax=ax)

    ax = axs[3]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*3:args.nlevs*4].T)
    ax.set_title('rho')
    fig.colorbar(c,ax=ax)

    fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True, sharey=True)
    ax = axs[0]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*4:args.nlevs*5].T)
    ax.set_title('xwind')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*5:args.nlevs*6].T)
    ax.set_title('ywind')
    fig.colorbar(c,ax=ax)

    ax = axs[2]
    c = ax.pcolor(xcopy.data.numpy()[:,args.nlevs*6:args.nlevs*7].T)
    ax.set_title('zwind')
    fig.colorbar(c,ax=ax)

    fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True)
    ax = axs[0]
    c = ax.plot(xcopy.data.numpy()[:,args.nlevs*7:(args.nlevs*7)+1][:,0])
    ax.set_title('shf')

    ax = axs[1]
    
    c = ax.plot(xcopy.data.numpy()[:,(args.nlevs*7)+1:(args.nlevs*7)+2][:,0])
    ax.set_title('lhf')

    ax = axs[2]
    c = ax.plot(xcopy.data.numpy()[:,(args.nlevs*7)+2:(args.nlevs*7)+3][:,0])
    ax.set_title('swtoa')
    
    plt.show()    

def scm_diff(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    recursion_indx = list(range(args.nlevs))
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext = yt_split['qtot_next'][:-1]
    qnext_inv = yt_inverse_split['qtot_next'][:-1]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    x_inv_split = nn_data.split_data(x,xyz='x')
    qtot_inv = x_inv_split['qtot'][:-1]
    
    output = []
    output.append(qnext[0])
    # xcopy = x.clone()
    difflist = []
    # args.xvar_multiplier = [1000., 1.]
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        # print(vdiff.shape)
        difflist.append(vdiff[:-1])
    xcopy = torch.cat(difflist, dim=1)
    diff_output = []
    qdiff = difflist[0]
    diff_output.append((qdiff[0]))
   
    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qdiff)):
        # Difference prediction
        # xcopy[t,:args.nlevs] = diff_output[t-1]
        # xcopy[t,args.nlevs:] = tdiff[t-1]
        yp_ = model(xcopy[t-1])
        # yp = model(torch.rand(xcopy[t-1].shape)/100.)
        qnew = output[t-1] + yp_[:args.nlevs]/1000.
        negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        output.append(qnew)
        diff_output.append(yp_)
        xcopy[t,:args.nlevs] = yp_

        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", qnext[t,:])

    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    yp = torch.stack(output)
    # yp = torch.from_numpy(qnext_ml)
    # yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    yp_inverse = yp
    yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            # 'qtot':qtot.data.numpy(),
            # 'qtot_next':qnext.data.numpy(), 
            # 'qtot_next_ml':qnext_ml,
            # 'theta_next':tnext_inv.data.numpy(), 
            # 'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def scm_diff_q_diag(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext_inv = yt_inverse_split['qtot'][1:]

    qtot_inv = yt_inverse_split['qtot'][:-1]
    
    output = []
    output.append(qtot_inv[0])
    difflist = []
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        difflist.append(vdiff[:-1])
    xcopy = torch.cat(difflist, dim=1)
    diff_output = []
    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qnext_inv)-1):
        # Difference prediction
        yp_ = model(xcopy[t])
        qnew = output[t-1] + yp_[:args.nlevs]/1000.
        negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        output.append(qnew)
        diff_output.append(yp_)
        # xcopy[t,:args.nlevs] = yp_

        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", qnext[t,:])

    yp = torch.stack(output)
    yp_inverse = yp
    yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    qnext_ml_inv = yp_inverse_split['qtot']

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)




def scm_diff_t(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    recursion_indx = list(range(args.nlevs))
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    # qnext = yt_split['qtot_next'][:-1]
    # qnext_inv = yt_inverse_split['qtot_next'][:-1]
    tnext = yt_split['theta_next'][:-1]
    tnext_inv = yt_inverse_split['theta_next'][:-1]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    x_inv_split = nn_data.split_data(x,xyz='x')
    # qtot_inv = x_inv_split['qtot'][:-1]
    theta_inv = x_inv_split['theta'][:-1]
    # output = []
    # output.append(qnext[0])
    toutput = []
    toutput.append(tnext[0])
    # xcopy = x.clone()
    difflist = []
    # args.xvar_multiplier = [1000., 1.]
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        # print(vdiff.shape)
        difflist.append(vdiff[:-1])
    xdiff = torch.cat(difflist, dim=1)
    xcopy = xdiff.clone()
    diff_output = []
    # qdiff = difflist[0]
    # diff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = difflist[1]
    tdiff_output.append((tdiff[0]))

    # plotX(xcopy[:10])
    # sys.exit(0)

    for t in range(1,len(tdiff)):
        # print(t)
        # Difference prediction
        # xcopy[t,:args.nlevs] = diff_output[t-1]
        # xcopy[t,args.nlevs:] = tdiff[t-1]
        yp_ = model(xcopy[t-1])
        # yp = model(torch.rand(xcopy[t-1].shape)/100.)
        # qnew = output[t-1] + yp_[:args.nlevs]/1000.
        tnew = toutput[t-1] + yp_[:args.nlevs]
        # print("tdiff", yp_[:args.nlevs])
        # print("x", xdiff[t,args.nlevs:args.nlevs*2])
        # print("qdiff", yp_[:args.nlevs])
        # negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        # if len(negative_values[0]) > 0:
            # qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        # output.append(qnew)
        toutput.append(tnew)
        # diff_output.append(yp_[:args.nlevs])
        tdiff_output.append(yp_[:args.nlevs])
        xcopy[t,args.nlevs:args.nlevs*2] = yp_
        # print("xcopy", xcopy[t,:args.nlevs])
        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", qnext[t,:])
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    # qp = torch.stack(output)
    tp = torch.stack(toutput)
    # print(tnext_inv.data.numpy()[:100,0])
    # print(tp[:200,0])
    ml_diff = tnext_inv.data.numpy()[:200,0] - tp.data.numpy()[:200,0] 
    per_diff = theta_inv.data.numpy()[0,0] - tnext_inv.data.numpy()[:200,0]
    # print(ml_diff)
    # yp = torch.from_numpy(qnext_ml)
    # yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    # yp_inverse = yp
    # yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    # qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']
    # qnext_ml_inv = qp
    tnext_ml_inv = tp

    output = {
            # 'qtot_next':qnext_inv.data.numpy(), 
            # 'qtot_next_ml':qnext_ml_inv.data.numpy(),
            # 'qtot':qtot_inv.data.numpy(),
            # 'qtot':qtot.data.numpy(),
            # 'qtot_next':qnext.data.numpy(), 
            # 'qtot_next_ml':qnext_ml,
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

    import IPython
    IPython.embed()

def scm_diff_t_diag(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    tnext_inv = yt_inverse_split['theta'][1:]

    theta_inv = yt_inverse_split['theta'][:-1]
    toutput = []
    toutput.append(theta_inv[0])
    # xcopy = x.clone()
    difflist = []
    # args.xvar_multiplier = [1000., 1.]
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        # print(vdiff.shape)
        difflist.append(vdiff[:-1])
    xdiff = torch.cat(difflist, dim=1)
    xcopy = xdiff.clone()

    # plotX(xcopy[:10])
    # sys.exit(0)
    for t in range(1,len(tnext_inv)-1):
        # print(t)
        # Difference prediction
        # xcopy[t,:args.nlevs] = diff_output[t-1]
        # xcopy[t,args.nlevs:] = tdiff[t-1]
        yp_ = model(xcopy[t])
        # yp = model(torch.rand(xcopy[t-1].shape)/100.)
        # qnew = output[t-1] + yp_[:args.nlevs]/1000.
        tnew = toutput[t-1] + yp_[:args.nlevs]/10.
        # print("tdiff", yp_[:args.nlevs])
        # print("x", xdiff[t,args.nlevs:args.nlevs*2])
        # print("qdiff", yp_[:args.nlevs])
        # negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        # if len(negative_values[0]) > 0:
            # qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        # output.append(qnew)
        toutput.append(tnew)
        # print("xcopy", xcopy[t,:args.nlevs])
        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", tnext[t,:])
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    # qp = torch.stack(output)
    tp = torch.stack(toutput)
    # print(tnext_inv.data.numpy()[:100,0])
    # print(tp[:200,0])
    ml_diff = tnext_inv.data.numpy()[:200,0] - tp.data.numpy()[:200,0] 
    per_diff = theta_inv.data.numpy()[0,0] - tnext_inv.data.numpy()[:200,0]
   
    tnext_ml_inv = tp

    output = {
            'theta':theta_inv.data.numpy(),
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

    # import IPython
    # IPython.embed()

def scm_diff_qt(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    recursion_indx = list(range(args.nlevs))
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext = yt_split['qtot_next'][:-1]
    qnext_inv = yt_inverse_split['qtot_next'][:-1]
    tnext = yt_split['theta_next'][:-1]
    tnext_inv = yt_inverse_split['theta_next'][:-1]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    x_inv_split = nn_data.split_data(x,xyz='x')
    qtot_inv = x_inv_split['qtot'][:-1]
    
    output = []
    output.append(qnext[0])
    toutput = []
    toutput.append(tnext[0])
    # xcopy = x.clone()
    difflist = []
    # args.xvar_multiplier = [1000., 1.]
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        # print(vdiff.shape)
        difflist.append(vdiff[:-1])
    xcopy = torch.cat(difflist, dim=1)
    diff_output = []
    qdiff = difflist[0]
    diff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = difflist[1]
    tdiff_output.append((tdiff[0]))

    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qdiff)):
        # print(t)
        # Difference prediction
        # xcopy[t,:args.nlevs] = diff_output[t-1]
        # xcopy[t,args.nlevs:] = tdiff[t-1]
        yp_ = model(xcopy[t-1])
        # yp = model(torch.rand(xcopy[t-1].shape)/100.)
        qnew = output[t-1] + yp_[:args.nlevs]/1000.
        tnew = toutput[t-1] + yp_[args.nlevs:]
        # print("tdiff", yp_[args.nlevs:])
        # print("qdiff", yp_[:args.nlevs])
        negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        output.append(qnew)
        toutput.append(tnew)
        diff_output.append(yp_[:args.nlevs])
        tdiff_output.append(yp_[args.nlevs:])
        xcopy[t,:(args.nlevs*2)] = yp_
        print("xcopy", xcopy[t,:(args.nlevs*2)])
        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", qnext[t,:])
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qp = torch.stack(output)
    tp = torch.stack(toutput)
    print(tnext_inv.data.numpy()[:100,0])
    print(tp[:100,0])
    # yp = torch.from_numpy(qnext_ml)
    # yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    # yp_inverse = yp
    # yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    # qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']
    qnext_ml_inv = qp
    tnext_ml_inv = tp

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            # 'qtot':qtot.data.numpy(),
            # 'qtot_next':qnext.data.numpy(), 
            # 'qtot_next_ml':qnext_ml,
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

def scm_diff_qt_diag(model, datasetfile, args):
    """
    SCM type run with qt_next prediction model
    """
    nn_data = data_io.Data_IO_validation(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'],
                        xvars=args.xvars,
                        yvars=args.yvars,
                        yvars2=args.yvars2,
                        add_adv=False, 
                        no_norm = True)
    print(args.xvars)
    recursion_indx = list(range(args.nlevs))
    x,y,y2,xmean,xstd,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # model = set_model(args)
    x_split = nn_data.split_data(x,xyz='x')
    yt_split = nn_data.split_data(y, xyz='y')
    # yt_inverse = nn_data._inverse_transform(y,ymean,ystd)
    yt_inverse = y
    yt_inverse_split = nn_data.split_data(yt_inverse, xyz='y')
    qnext_inv = yt_inverse_split['qtot'][1:]
    tnext_inv = yt_inverse_split['theta'][1:]

    # x_inv_split = nn_data._inverse_transform_split(x_split,xmean,xstd,xyz='x')
    x_inv_split = nn_data.split_data(x,xyz='x')
    qtot_inv = yt_inverse_split['qtot'][:-1]
    theta_inv = yt_inverse_split['theta'][:-1]

    output = []
    output.append(qtot_inv[0])
    toutput = []
    toutput.append(theta_inv[0])
    # xcopy = x.clone()
    difflist = []
    # args.xvar_multiplier = [1000., 1.]
    for v,m in zip(args.xvars, args.xvar_multiplier):
        vdiff = (x_split[v][1:] - x_split[v][:-1])*m
        # print(vdiff.shape)
        difflist.append(vdiff[:-1])
    xcopy = torch.cat(difflist, dim=1)
    diff_output = []
    qdiff = difflist[0]
    diff_output.append((qdiff[0]))
   
    tdiff_output = []
    tdiff = difflist[1]
    tdiff_output.append((tdiff[0]))

    # plotX(x)
    # sys.exit(0)

    for t in range(1,len(qnext_inv)-1):
        # print(t)
        yp_ = model(xcopy[t])
        qnew = output[t-1] + yp_[:args.nlevs]/1000.
        tnew = toutput[t-1] + yp_[args.nlevs:]/10.
        print("tdiff", yp_[args.nlevs:])
        print("qdiff", yp_[:args.nlevs])
        negative_values = torch.where(qnew<0.)
        # print(negative_values[0])
        if len(negative_values[0]) > 0:
            qnew[negative_values] = 1.e-6 #output[t-1][negative_values]
            # yp_[negative_values] = diff_output[t-1][negative_values]
        output.append(qnew)
        toutput.append(tnew)
        diff_output.append(yp_[:args.nlevs])
        tdiff_output.append(yp_[args.nlevs:])
        # print("xcopy", xcopy[t,:(args.nlevs*2)])
        # print("Diff predict", diff_output[t][0:10])
        # print("Diff true", (qnext[t][0:10] - qnext[t-1][0:10])*1000.)
        # print("Predict", output[t][:])
        # print("True", qnext[t,:])
    # yp = torch.from_numpy(np.concatenate([qnext_ml, tnext_ml], axis=1))
    qp = torch.stack(output)
    tp = torch.stack(toutput)
    # yp = torch.from_numpy(qnext_ml)
    # yp_inverse = nn_data._inverse_transform(yp, ymean, ystd)
    # yp_inverse = yp
    # yp_inverse_split = nn_data.split_data(yp_inverse, xyz='y')
    # qnext_ml_inv = yp_inverse_split['qtot_next']
    # tnext_ml_inv = yp_inverse_split['theta_next']
    qnext_ml_inv = qp
    tnext_ml_inv = tp

    output = {
            'qtot_next':qnext_inv.data.numpy(), 
            'qtot_next_ml':qnext_ml_inv.data.numpy(),
            'qtot':qtot_inv.data.numpy(),
            # 'qtot':qtot.data.numpy(),
            # 'qtot_next':qnext.data.numpy(), 
            # 'qtot_next_ml':qnext_ml,
            'theta_next':tnext_inv.data.numpy(), 
            'theta_next_ml':tnext_ml_inv.data.numpy()
    }
    hfilename = args.model_name.replace('.tar','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
    model_loc = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = model_loc+"tdiff_006_lyr_333_in_055_out_0388_hdn_050_epch_00150_btch_023001AQTS_mse_023001AQT_normalise_stkd_tanh.tar"
    datasetfile = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_020.hdf5"
    # normaliser_region = "023001AQT_normalise_60_glb"
    # normaliser_region = "023001AQT_standardise_mx"
    normaliser_region = "023001AQT_normalise"
    data_region = "0N100W"
    args = set_args(model_file, normaliser_region, data_region)
    model = set_model(model_file, args)
    # scm(model, datasetfile, args)
    # scm_diff(model, datasetfile, args)
    # scm_diff_qt(model, datasetfile, args)
    # scm_diff_t(model, datasetfile, args)
    # scm_diff_q_diag(model, datasetfile, args)
    scm_diff_t_diag(model, datasetfile, args)
    # scm_diff_qt_diag(model, datasetfile, args)