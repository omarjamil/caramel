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
    optimizer =  torch.optim.Adam(model.parameters(), lr=1.e-3)
    # optimizer =  torch.optim.SGD(mlp.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # loss_function = torch.nn.MSELoss()
    return optimizer, scheduler

def kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def reconstruction_loss(recon_x,x):
    l1_loss = torch.nn.functional.l1_loss
    MAE = l1_loss(recon_x,x,reduction='mean')
    return MAE

def loss_function(recon_x, x, mu, logvar):
    
    recon_loss = reconstruction_loss(recon_x, x)
    kld = kld_loss(mu, logvar)

    return recon_loss, kld

def training_step(batch, batch_, batch_idx, model, optimizer, device, input_indices=None):
    """
    """
    x,y,x2 = batch
    x_, y_  = batch_
    x_ = torch.cat(x_,dim=1).to(device)
    y_ = torch.cat(y_,dim=1).to(device)
    decoded, mu, logvar = model(x_)
    # print(decoded[:3,0],y_[:3,0])
    recon_loss, kld = loss_function(decoded,x_, mu, logvar)
    loss = recon_loss + kld
    # qloss = loss_function(1000.*qpredict,1000.*qout, reduction='mean')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_, batch_idx, model, device, input_indices=None):
    """
    """
    x,y,x2 = batch
    # qin = (x[0][list(range(1,len(x[0])-1,2)),:]).to(device)
    # qout = (x[0][list(range(2,len(x[0]),2)),:]).to(device)
    x_,y_ = batch_
    x_ = torch.cat(x_,dim=1).to(device)
    y_ = torch.cat(y_,dim=1).to(device)
    with torch.no_grad():
        decoded, mu, logvar = model(x_)
        # print(decoded[:3,0],y_[:3,0])
        recon_loss, kld = loss_function(decoded,x_, mu, logvar)
        loss = recon_loss + kld
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
    mlp = model.VAE(args.in_features, args.latent_size)
    pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Number of traninable parameter: {0}".format(pytorch_total_params))
   
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
    
    return mlp, optimizer, scheduler

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


def create_diff_inout_vars(batch, xvar_multiplier, yvar_multiplier, xstoch=False):
    x,y,x2 = batch
    xlist = []
    xdifflist = []

    for v,m in zip(x, xvar_multiplier):
        vdiff = (v[1:] - v[:-1])*m
        xdifflist.append(vdiff)
    
    ylist = []
    ydifflist = []
    for v,m in zip(y,yvar_multiplier):
        vdiff = (v[1:] - v[:-1])*m
        ydifflist.append(vdiff)

    if xstoch:
        for i,v in enumerate(xdifflist):
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
        ylist.append(v[list(range(0,len(v)-1,1))])

    return (xlist, ylist)

def train_loop(model, optimizer, scheduler, args):
    
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
            batch_ = create_diff_inout_vars(batch, args.xvar_multiplier, args.yvar_multiplier, xstoch=args.xstoch)
            model.train()
            
            loss = training_step(batch, batch_, batch_idx, model, optimizer, args.device, input_indices=input_indices)
            
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
            batch_ = create_diff_inout_vars(batch, args.xvar_multiplier, args.yvar_multiplier)
            model.eval()
            loss = validation_step(batch, batch_, batch_idx, model, args.device, input_indices=input_indices)
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