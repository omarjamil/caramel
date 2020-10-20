import torch
import pickle
import numpy as np
import math
import sys
import os
import h5py
import argparse

import model
# import data_io_stacked as data_io
import data_io_batched as data_io

def minkowski_error(prediction, target, minkowski_parameter=1.5):
    """
    Minkowski error to be better deal with outlier errors
    """
    error = torch.abs(prediction - target)**minkowski_parameter
    # print(error.shape)
    # print(error)
    loss = torch.mean(error)
    return loss

def configure_optimizers(model):
    optimizer =  torch.optim.Adam(model.parameters(), lr=1.e-4)
    # optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # loss_function = torch.nn.MSELoss()
    return optimizer, scheduler

def training_step(batch, batch_, batch_idx, model, loss_function, optimizer, device, input_indices=None):
    """
    """
    x,y,x2 = batch
    # qin = (x[0][list(range(1,len(x[0])-1,2)),:]).to(device)
    # qout = (x[0][list(range(2,len(x[0]),2)),:]).to(device)
    x_, y_  = batch_
    # print(y_)
    x_ = torch.cat(x_,dim=1).to(device)
    y_ = torch.cat(y_,dim=1).to(device)
    # print("x", torch.mean(x_[0:10,:55],dim=0))
    # print("y", torch.mean(y_[0:10,:55],dim=0))
    # if torch.where(y_>20.)[0].numpy().any():
        # print(torch.where(y_>1.))
    # if torch.where(y_<-20.)[0].numpy().any():
        # print(torch.where(y_<-1.))    
    # print(len(qin), len(qout), len(y_))
    output = model(x_)
    # print("ML", output)
    # print("truth", y_)
    # print(y_.shape)
    # print(y_.shape[1]//2)
    # qpredict = qin+(output[:,:(y_.shape[1]//2)]/1000.) 
    # print(qout - qpredict)
    # print(x_.shape, y_.shape, output.shape)
    # print("In", x_[0,55:110])
    # print("True", y_[0,:])
    # print("Pred", output[0,:])

    loss = loss_function(output,y_, reduction='none')
    diff_loss = torch.sum(torch.mean(loss,dim=0))
    
    # diff_loss = loss_function(output,y_, reduction='mean')
    # print(diff_loss)
    # qloss = loss_function(1000.*qpredict,1000.*qout, reduction='mean')
    optimizer.zero_grad()
    # if torch.lt(qpredict,0.).any():
        # print("-",end=' ')
        # print(diff_loss.item(), qloss.item())
        # loss = diff_loss + 1000.*qloss
    # else:
        # loss = diff_loss + qloss
    # loss = diff_loss + qloss
    loss = diff_loss
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_, batch_idx, model, loss_function, device, input_indices=None):
    """
    """
    x,y,x2 = batch
    # qin = (x[0][list(range(1,len(x[0])-1,2)),:]).to(device)
    # qout = (x[0][list(range(2,len(x[0]),2)),:]).to(device)
    x_,y_ = batch_
    x_ = torch.cat(x_,dim=1).to(device)
    y_ = torch.cat(y_,dim=1).to(device)
    with torch.no_grad():
        output = model(x_)
        # qpredict = qin+(output/1000.) 
        # print("True", y[0,0:5])
        # print("Pred", output[0,0:5])
        diff_loss = loss_function(output,y_, reduction='mean')
        # loss = loss_function(output,y_, reduction='mean')
        # diff_loss = torch.sum(torch.mean(loss,dim=0))
        # qloss = loss_function(1000.*qpredict,1000.*qout, reduction='mean')
        # loss = diff_loss + qloss
        loss = diff_loss
    return loss


def checkpoint_save(epoch: int, nn_model: model, nn_optimizer: torch.optim, training_loss: list, validation_loss: list, model_name: str, locations: dict, args):
    """
    Save model checkpoints
    """
    checkpoint_name = model_name.replace('.tar','_chkepo_{0}.tar'.format(str(epoch).zfill(3)))
    torch.save({'epoch':epoch,
                'model_state_dict':nn_model.state_dict(),
                'optimizer_state_dict':nn_optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                locations['model_loc']+'/'+checkpoint_name)


def set_model(args):
    # mlp = model.MLP(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    mlp = model.MLP_tanh(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size,scale=1.)
    # mlp = model.MLP_sig(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLP_BN_tanh(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLPSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # skipindx = list(range(args.nlevs))
    # mlp = model.MLPSubSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size, skipindx)
    # mlp = model.MLPDrop(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLP_BN(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.ResMLP(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Number of traninable parameter: {0}".format(pytorch_total_params))

    if args.loss == "mae":
        loss_function = torch.nn.functional.l1_loss #torch.nn.L1Loss()
    elif args.loss == "mse":
        loss_function = torch.nn.functional.mse_loss #torch.nn.MSELoss()
    elif args.loss == "mink":
        loss_function = minkowski_error
    elif args.loss == "huber":
        loss_function = torch.nn.functional.smooth_l1_loss
    optimizer, scheduler = configure_optimizers(mlp)

    if args.warm_start is not None:
    # Load the save model 
        checkpoint = torch.load(args.locations['model_loc']+'/'+args.warm_start, map_location=args.device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['training_loss']
        epoch = checkpoint['epoch']
        args.model_name = args.model_name.replace('.tar','_{0}.tar'.format("cont"))
    else:
        mlp.to(args.device)
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    
    return mlp, loss_function, optimizer, scheduler

def train_dataloader(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_dataset = data_io.ConcatDataset("train",args.nlevs, train_dataset_file, args.locations['normaliser_loc'], args.batch_size, xvars=args.xvars,
             yvars=args.yvars, xvars2=args.xvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm, fmin=args.fmin, fmax=args.fmax)
    indices = list(range(train_dataset.__len__()))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=None, batch_size=None, sampler=train_sampler, shuffle=False)
    return train_loader

def test_dataloader(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    test_dataset = data_io.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], args.batch_size, xvars=args.xvars,
             yvars=args.yvars, xvars2=args.xvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm, fmin=args.fmin, fmax=args.fmax)
    indices = list(range(test_dataset.__len__()))
    test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=None, batch_size=None, sampler=test_sampler, shuffle=False)
    return validation_loader


def create_diff_inout_vars(batch, xvar_multiplier, yvar_multiplier, train_on_x2=False, no_xdiff=False, no_ydiff=False, xstoch=True):
    x,y,x2 = batch
    xlist = []
    xdifflist = []

    if no_xdiff:
        for v,m in zip(x, xvar_multiplier):
            xdifflist.append(v[1:]*m)
    else:
        for v,m in zip(x, xvar_multiplier):
            vdiff = (v[1:] - v[:-1])*m
            xdifflist.append(vdiff)

    if train_on_x2:
        for v in (x2):
            xdifflist.append(v[1:])

    ylist = []
    ydifflist = []
    if no_ydiff:
        for v,m in zip(y, yvar_multiplier):
            ydifflist.append(v[1:]*m)
    else:
        for v,m in zip(y,yvar_multiplier):
            vdiff = (v[1:] - v[:-1])*m
            ydifflist.append(vdiff)

    # Stochastic perturbation of prognostic inputs
    # mean0 = torch.mean(xdifflist[0],dim=0)/5.
    # std0 = torch.std(xdifflist[0],dim=0)/10.
    # rand0 = torch.normal(mean0,std0)
    # mean1 = torch.mean(xdifflist[1],dim=0)/5.
    # std1 = torch.std(xdifflist[1],dim=0)/10.
    # rand1 = torch.normal(mean1,std1)
    # print("mean",mean)
    # print("std",std)
    # print("rand",rand)
    # print("before", xdifflist[0][0])
    # xdifflist[0] = xdifflist[0] + rand0
    # xdifflist[1] = xdifflist[1] + rand1

    # print("after", xdifflist[0][0])

    # Stochastic perturbation of all inputs
    if xstoch:
        for i,v in enumerate(xdifflist):
            # vmean = torch.mean(v,dim=0)/5.
            # vstd = torch.std(v,dim=0)/10.
            # vrand = torch.normal(vmean,vstd)
            if i == 0:
                vmean = torch.mean(v,dim=0)
                vstd = torch.std(v,dim=0)
                # vrand = torch.normal(vmean,vstd)/10.
                vrand = torch.normal(vmean,vstd)
                # print("var {0} mean {1} std {2} vrand {3}".format(v[0,:], vmean, vstd, vrand))
                v = v + vrand
            xlist.append(v[list(range(0,len(v)-1,1))])
    else:
        for v in xdifflist:
            # xlist.append(v[list(range(0,len(v)-1,2))])
            xlist.append(v[list(range(0,len(v)-1,1))])

    for v in ydifflist:
        # ylist.append(v[list(range(1,len(v),1))])
        ylist.append(v[list(range(0,len(v)-1,1))])


    # ylist = [difflist[0][list(range(1,len(difflist[0]),2))]]
    # ylist = [difflist[1][list(range(1,len(difflist[1]),2))]]
    # ylist = [difflist[0][list(range(1,len(difflist[0]),2))], difflist[1][list(range(1,len(difflist[1]),2))]]
    return (xlist, ylist)

def train_loop(model, loss_function, optimizer, scheduler, args):
    
    training_loss = []
    validation_loss = []
    train_ldr = train_dataloader(args)
    test_ldr = test_dataloader(args)
    input_indices = list(range(args.nlevs))
    for epoch in range(1, args.epochs + 1):
        ## Training
        train_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            # Sets the model into training mode
            batch_ = create_diff_inout_vars(batch, args.xvar_multiplier, args.yvar_multiplier, train_on_x2=args.train_on_x2, no_xdiff=args.no_xdiff, no_ydiff=args.no_ydiff, xstoch=args.xstoch)
            model.train()
            
            loss = training_step(batch, batch_, batch_idx, model, loss_function, optimizer, args.device, input_indices=input_indices)
            
            train_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                x,y, x2=batch
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(epoch, 
                batch_idx * len(x), len(train_ldr.dataset)*(args.batch_size/2.),100. * batch_idx / (len(train_ldr)),
                loss.item() / len(x)))
        average_loss = train_loss / len(train_ldr.dataset)
        print('====> Epoch: {} Average loss: {:.2e}'.format(epoch, average_loss))
        training_loss.append(average_loss)
        scheduler.step()
        ## Testing
        test_loss = 0
        for batch_idx, batch in enumerate(test_ldr):
            batch_ = create_diff_inout_vars(batch, args.xvar_multiplier, args.yvar_multiplier, train_on_x2=args.train_on_x2, no_xdiff=args.no_xdiff, no_ydiff=args.no_ydiff, xstoch=args.xstoch)
            model.eval()
            loss = validation_step(batch, batch_, batch_idx, model, loss_function, args.device, input_indices=input_indices)
            test_loss += loss.item()
        average_loss_val = test_loss / len(test_ldr.dataset) 
        print('====> validation loss: {:.2e}'.format(average_loss_val))
        validation_loss.append(average_loss_val)
        if epoch % 2 == 0:
            checkpoint_name = args.model_name.replace('.tar','_chkepo_{0}.tar'.format(str(epoch).zfill(3)))
            torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                args.locations['chkpnt_loc']+'/'+checkpoint_name)
            # checkpoint_save(epoch, model, optimizer, training_loss, validation_loss, args.model_name, args.locations, args)            
     # Save the final model
    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                args.locations['model_loc']+'/'+args.model_name)   
    return training_loss, validation_loss