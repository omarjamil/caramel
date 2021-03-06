import torch
import pickle
import numpy as np
import math
import sys
import os
import h5py
import argparse

import model
import data_io

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


def training_step(batch, batch_idx, model, loss_function, optimizer, device, train_on_y2=False):
    """
    """
    x, y, y2 = batch
    y = x.clone()
    x = x.to(device)
    if train_on_y2:
        y = y2.to(device)
    else:
        y = y.to(device)
    output = model(x)
    # print(output)
    loss = loss_function(output,y, reduction='mean')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_idx, model, loss_function, device, train_on_y2=False):
    """
    """
    x,y, y2 = batch
    y = x.clone()
    x = x.to(device)
    if train_on_y2:
        y = y2.to(device)
    else:
        y = y.to(device)
    with torch.no_grad():
        output = model(x)
        loss = loss_function(output, y, reduction='mean')
        # print("ouptut",output[0,:5])
        # print("y",y[0,:5])
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
    mlp = model.AE(args.in_features)
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

    if args.warm_start:
    # Load the save model 
        checkpoint = torch.load(args.locations['model_loc']+'/'+args.model_name, map_location=args.device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        args.model_name = args.model_name.replace('.tar','_{0}.tar'.format(args.region))
    else:
        mlp.to(args.device)
    print("Model's state_dict:")
    for param_tensor in mlp.state_dict():
        print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())
    
    return mlp, loss_function, optimizer, scheduler

def train_dataloader(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=True)
    return train_loader

def test_dataloader(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader



def train_loop(model, loss_function, optimizer, scheduler, args):
    
    training_loss = []
    train_ldr = train_dataloader(args)
    validation_loss = []
    test_ldr = test_dataloader(args)
    
    for epoch in range(1, args.epochs + 1):
        ## Training
        train_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            # Sets the model into training mode
            model.train()
            loss = training_step(batch, batch_idx, model, loss_function, optimizer, args.device, train_on_y2=args.train_on_y2)
            train_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                x,y, y2=batch
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(epoch, 
                batch_idx * len(x), len(train_ldr.dataset),100. * batch_idx / len(train_ldr),
                loss.item() / len(x)))
        average_loss = train_loss / len(train_ldr.dataset)
        print('====> Epoch: {} Average loss: {:.2e}'.format(epoch, average_loss))
        training_loss.append(average_loss)
        scheduler.step()
        ## Testing
        test_loss = 0
        for batch_idx, batch in enumerate(test_ldr):
            model.eval()
            loss = validation_step(batch, batch_idx, model, loss_function, args.device, train_on_y2=args.train_on_y2)
            test_loss += loss.item()
        average_loss_val = test_loss / len(test_ldr.dataset)
        print('====> validation loss: {:.2e}'.format(average_loss_val))
        validation_loss.append(average_loss_val)
        if epoch % 100 == 0:
            checkpoint_save(epoch, model, optimizer, training_loss, validation_loss, args.model_name, args.locations, args)            
     # Save the final model
    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'training_loss':training_loss,
                'validation_loss':validation_loss,
                'arguments':args},
                args.locations['model_loc']+'/'+args.model_name)   
    return training_loss, validation_loss