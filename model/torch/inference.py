import torch
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py
import argparse

import model
import data_io
import normalize



parser = argparse.ArgumentParser(description='Train Q')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--with-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--data-region', type=str, help='data region')
parser.add_argument('--normaliser', type=str, help='normaliser region')
parser.add_argument('--identifier', type=str, 
                    help='Added to model name as a unique identifier;  also needed for warm start from a previous model')

parser.add_argument('--nhdn-layers', type=int, default=6, metavar='N',
                    help='Number of hidden layers (default: 6)')

args = parser.parse_args()
args.cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
region = args.data_region

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
            "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch/",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# region = "50S69W"

# Data normalizer class
nt = normalize.Normalizers(locations)
# Training and testing data class
# nn_data = data_io.Data_IO(region, locations)
nn_data = data_io.Data_IO_validation(region, locations)
nn_data.get_data(0)

# Define the Model
# n_inputs,n_outputs=140,70
in_features, nb_classes=123,30
nb_hidden_layer = args.nhdn_layers 
hidden_size = 256
mlp = model.MLP(in_features, nb_classes, nb_hidden_layer, hidden_size)
# mlp = model.MLP_BN(in_features, nb_classes, nb_hidden_layer, hidden_size)
model_name = "q_qadv_t_tadv_swtoa_lhf_shf_qphys_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_mse_vlr.tar".format(str(nb_hidden_layer).zfill(3),
                                                                                    str(in_features).zfill(3),
                                                                                    str(nb_classes).zfill(3),
                                                                                    str(hidden_size).zfill(4),
                                                                                    str(args.epochs).zfill(3),
                                                                                    str(args.batch_size).zfill(5),
                                                                                    args.identifier)
optimizer =  torch.optim.Adam(mlp.parameters())
# loss_function = torch.nn.MSELoss()

# Load the save model 
print("Loading PyTorch model: {0}".format(locations['model_loc']+'/'+model_name))
checkpoint = torch.load(locations['model_loc']+'/'+model_name, map_location=device)
mlp.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# loss = checkpoint['training_loss']
# epoch = checkpoint['epoch']
mlp.eval() 
# or to continue traininig
# mlp.train()
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in mlp.state_dict():
    print(param_tensor, "\t", mlp.state_dict()[param_tensor].size())

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])





def predict_q(q_in, nlevs):
    """
    Test the humidity model
    """
    global mlp
    
    qphys_predict = mlp(torch.from_numpy(q_in))
        
    # qphys_predict_denorm = nt.inverse_minmax(qphys_predict.data.numpy(), qphys_scale.data.numpy(), qphys_feature_min.data.numpy(), qphys_feature_max.data.numpy(), qphys_data_min.data.numpy())
    qphys_predict_denorm = nt.inverse_std(qphys_predict.data.numpy(), nt.qphys_stdscale.data.numpy()[...,:nlevs], nt.qphys_mean.data.numpy()[...,:nlevs])
    # qphys_predict_denorm = nt.inverse_std(qphys_predict.data.numpy(), nt.qphys_dot_stdscale_s.data.numpy(), nt.qphys_dot_mean_s.data.numpy())
    
    return qphys_predict_denorm.reshape(nlevs)

def q_scm(region='50S69W'):
    """
    Using qphys prediction and then feeding them back into the emulator, 
    create a timeseries of prediction to compare with validation data
    """
    
    start = 0
    end = 200
    nlevs = 30
    # q_tot = nn_data.q_tot_test[start:end,:nlevs]
    q_inv =  nt.inverse_std(nn_data.q_tot_test[start:end,:nlevs], nt.q_stdscale.data.numpy()[...,:nlevs], nt.q_mean.data.numpy()[...,:nlevs])
    qphys = nn_data.qphys_test[start:end,:nlevs]
    qadv_dot = nn_data.q_tot_adv_test[start:end,:nlevs]
    qadv_dot_inv = nt.inverse_std(qadv_dot, nt.qadv_stdscale.data.numpy()[...,:nlevs], nt.qadv_mean.data.numpy()[...,:nlevs])

    qadv_dot_inv = qadv_dot_inv * 600.
    qt_test  = np.concatenate((nn_data.q_tot_test[start:end,:nlevs], nn_data.q_tot_adv_test[start:end,:nlevs], nn_data.theta_test[start:end,:nlevs], nn_data.theta_adv_test[start:end,:nlevs], nn_data.sw_toa_test[start:end], nn_data.lhf_test[start:end], nn_data.shf_test[start:end]),axis=1)
    # qt_test  = np.concatenate((nn_data.q_tot_test[:,:nlevs], nn_data.theta_test[:,:nlevs], nn_data.sw_toa_test, nn_data.lhf_test, nn_data.shf_test),axis=1)
    
    qphys_inv = nt.inverse_std(qphys, nt.qphys_stdscale.data.numpy()[...,:nlevs], nt.qphys_mean.data.numpy()[...,:nlevs])

    n_steps = len(range(start,end))
    
    q_start = q_inv[0,:]
    q_ml = np.zeros((n_steps,nlevs))
    q_sane = np.zeros((n_steps,nlevs))
    qphys_drift = np.zeros((n_steps,nlevs))
    qphys_pred_drift = np.zeros((n_steps,nlevs))
    qphys_ml = np.zeros((n_steps,nlevs))
    q_ml[0,:] = q_start
    q_sane[0,:] = q_start
    qphys_drift[0,:] = qphys_inv[0,:]
    qphys_pred_drift[0,:] = qphys_inv[0,:]
    qphys_ml[0,:] = qphys_inv[0,:]

    printProgressBar(0, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for t_step in range(1,n_steps):
        
        # q_in = (qcomb_dot_test[t_step-1,:]).reshape((1,140))
        q_in = (qt_test[t_step,:]).reshape((1,in_features))
        # q_in = (qcomb_dot_test[t_step,:]).reshape((1,70))
        qphys_pred = predict_q(q_in, nlevs)
        qphys_ml[t_step,:] = qphys_pred
        #q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_inv[t_step,:] + qphys_pred
        # q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_pred
        q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_ml[t_step-1,:]
        q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_inv[t_step-1, :]
        # q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_dot_inv[t_step,:] + qphys_inv[t_step]
        qphys_drift[t_step,:] = qphys_drift[t_step-1,:] + qphys_inv[t_step]
        qphys_pred_drift[t_step,:] = qphys_pred_drift[t_step-1,:] + qphys_pred
        printProgressBar(t_step, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # qadv_un = qadv_inv #qadv_normaliser.inverse_transform(qadv_norm)
    output_dict = {"q_ml":q_ml[:],"q":q_inv[:],"qphys_ml":qphys_ml,"qphys":qphys_inv[:],"qadv_dot":qadv_dot[:], "qadv_dot_inv":qadv_dot_inv[:], "q_sane":q_sane[:], "qphys_drift":qphys_drift[:], "qphys_pred_drift":qphys_pred_drift[:]}
    outfile_name = model_name.replace('.tar','.hdf5')    
    with h5py.File(outfile_name,'w') as outfile:
        for k,v in output_dict.items():
            outfile.create_dataset(k,data=v)

def q_inference(region='50S69W'):
    """
    q model inference/testing
    """
    start=0
    end=200
    nlevs=30
    # Testing data
    qphys_norm_test = nn_data.qphys_test[start:end,:nlevs]
    # qphys_norm_test = nn_data.qphys_train[start:end,:nlevs]
    # qcomb_test  = np.concatenate((nn_data.qadv_norm_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = np.concatenate((nn_data.qadd_dot_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = nn_data.qadd_dot_test
    # qcomb_dot_test  = np.concatenate((nn_data.q_norm_test_s, nn_data.qadv_dot_norm_test_s),axis=1)
    qt_test  = np.concatenate((nn_data.q_tot_test[start:end,:nlevs], nn_data.q_tot_adv_test[start:end,:nlevs], nn_data.theta_test[start:end,:nlevs], nn_data.theta_adv_test[start:end,:nlevs], nn_data.sw_toa_test[start:end,:], nn_data.lhf_test[start:end,:], nn_data.shf_test[start:end,:]),axis=1)
    # qt_test  = np.concatenate((nn_data.q_tot_test[start:end,:nlevs], nn_data.theta_test[start:end,:nlevs], nn_data.sw_toa_test[start:end,:], nn_data.lhf_test[start:end,:], nn_data.shf_test[start:end,:]),axis=1)
    # qt_test  = np.concatenate((nn_data.q_tot_train[start:end,:nlevs], nn_data.q_tot_adv_train[start:end,:nlevs], nn_data.theta_train[start:end,:nlevs], nn_data.theta_adv_train[start:end,:nlevs], nn_data.sw_toa_train[start:end,:], nn_data.lhf_train[start:end,:], nn_data.shf_train[start:end,:]),axis=1)
    
    #train_in, train_out = qcomb_dot_train, qphys_norm_train
    #test_in, test_out = qcomb_dot_test, qphys_norm_test
    x_t,y_t = torch.from_numpy(qt_test[:]), qphys_norm_test[:]

    prediction = mlp(x_t)
    # qphys_predict_denorm = nt.inverse_minmax_tensor(prediction, qphys_scale, qphys_feature_min, qphys_feature_max, qphys_data_min)
    # qphys_test_denorm = nt.inverse_minmax_tensor(torch.from_numpy(qphys_norm_test[:]), qphys_scale, qphys_feature_min, qphys_feature_max, qphys_data_min)
    qphys_predict_denorm = nt.inverse_std(prediction, nt.qphys_stdscale[...,:nlevs], nt.qphys_mean[...,:nlevs])
    qphys_test_denorm = nt.inverse_std(torch.from_numpy(qphys_norm_test[:]), nt.qphys_stdscale[...,:nlevs], nt.qphys_mean[...,:nlevs])
    # qphys_predict_denorm = prediction
    # qphys_test_denorm = torch.from_numpy(qphys_norm_test)

    hfilename = model_name.replace('.tar','_qphys.hdf5')
    output={'qphys_predict':qphys_predict_denorm.data.numpy(),'qphys_test':qphys_test_denorm.data.numpy()}
    
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
    q_inference(region=region)
    q_scm(region=region)