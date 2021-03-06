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

def recursive_training_step(batch, batch_idx, model, loss_function, optimizer, device, recur_input_indx):
    """
    recur_input_indx: which of the inputs are to replaced by model output from the previous step 
    """
    # print("In teacher force mode")
    x, y, y2 = batch
    x = x.to(device)
    y = y.to(device)
    output = []
    output.append(model(x[0]))
    for i in range(1,len(x)):
        xmod = x.clone()
        xmod[i,recur_input_indx] = output[i-1]
        # print(xmod[i,:5], x[i,:5])
        output.append(model(xmod[i]))
    batched_output = torch.stack(output)
    # print(batched_output.shape, y.shape)
    loss = loss_function(batched_output, y, reduction='mean')
    # print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def training_step(batch, batch_idx, model, loss_function, optimizer, device, input_indices=None):
    """
    """
    x, y, y2 = batch
    if input_indices is not None:
        y = (y[0] - x[0][...,input_indices])*100.
        x = torch.cat(x,dim=1).to(device)
        y = y.to(device)
    else:
        x = torch.cat(x,dim=1).to(device)
        y = y.to(device)
    output = model(x)
    # print("Pred", output[0,0:5])
    # loss = loss_function(output,y, reduction='mean')
    diff_loss = loss_function(output,y, reduction='none')
    loss = torch.sum(torch.mean(loss,dim=0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation_step(batch, batch_idx, model, loss_function, device, input_indices=None):
    """
    """
    x,y, y2 = batch
    if input_indices is not None:
        y = (y - x[0][...,input_indices])*100.
        x = torch.cat(x,dim=1).to(device)
        y = y.to(device)
    else:
        x = torch.cat(x,dim=1).to(device)
        y = y.to(device)
    with torch.no_grad():
        output = model(x)
        # print("True", y[0,0:5])
        # print("Pred", output[0,0:5])
        loss = loss_function(output, y, reduction='mean')
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
    mlp = model.MLP_tanh(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLPSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # skipindx = list(range(args.nlevs))
    # mlp = model.MLPSubSkip(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size, skipindx)
    # mlp = model.MLPDrop(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
    # mlp = model.MLP_BN(args.in_features, args.nb_classes, args.nb_hidden_layers, args.hidden_size)
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
             yvars=args.yvars, yvars2=args.yvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm)
    indices = list(range(train_dataset.__len__()))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=None, batch_size=None, sampler=train_sampler, shuffle=False)
    return train_loader

def test_dataloader(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    test_dataset = data_io.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], args.batch_size, xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm)
    indices = list(range(test_dataset.__len__()))
    test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=None, batch_size=None, sampler=test_sampler, shuffle=False)
    return validation_loader



def train_loop(model, loss_function, optimizer, scheduler, args):
    
    training_loss = []
    validation_loss = []
    recur_input_indx = list(range(args.nlevs))
    train_ldr = train_dataloader(args)
    test_ldr = test_dataloader(args)
    recursive_train_interval = 1.1
    input_indices = list(range(args.nlevs))
    for epoch in range(1, args.epochs + 1):
        ## Training
        train_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            # Sets the model into training mode
            model.train()
            if batch_idx%1 == recursive_train_interval:
                loss = recursive_training_step(batch, batch_idx, model, loss_function, optimizer, args.device, recur_input_indx)
            else:
                loss = training_step(batch, batch_idx, model, loss_function, optimizer, args.device, input_indices=input_indices)
            
            train_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                x,y, y2=batch
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(epoch, 
                batch_idx * len(x), len(train_ldr.dataset)*args.batch_size,100. * batch_idx / len(train_ldr),
                loss.item() / len(x)))
        average_loss = train_loss / len(train_ldr.dataset)
        print('====> Epoch: {} Average loss: {:.2e}'.format(epoch, average_loss))
        training_loss.append(average_loss)
        scheduler.step()
        ## Testing
        test_loss = 0
        for batch_idx, batch in enumerate(test_ldr):
            model.eval()
            loss = validation_step(batch, batch_idx, model, loss_function, args.device, input_indices=input_indices)
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
                args.locations['model_loc']+'/'+checkpoint_name)
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