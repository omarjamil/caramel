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

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
            "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch/",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

parser = argparse.ArgumentParser(description='Train Q')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--with-cuda', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()
args.cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

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

region = "50S69W"
# Data normalizer class
nt = normalize.Normalizers(locations)
# Training and testing data class
nn_data = data_io.Data_IO(region, locations)

# the Model
# n_inputs,n_outputs=140,70
n_inputs,n_outputs=140,70
n_layers = 6
mlp = model.MLP_06(n_inputs,n_outputs)
optimizer =  torch.optim.Adam(mlp.parameters())
loss_function = torch.nn.MSELoss()

# Load the save model 
model_fname = "qloss_0_qphys_dot_1_{1}deep_epoch_{0}_qcomb_std_Smth.tar".format(str(args.epochs).zfill(3), str(n_layers).zfill(2))
checkpoint = torch.load(locations['model_loc']+'/'+model_fname, map_location=device)
mlp.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']
epoch = checkpoint['epoch']
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





def predict_q(q_in):
    """
    Test the humidity model
    """
    global mlp
    
    qphys_predict = mlp(torch.from_numpy(q_in))
        
    # qphys_predict_denorm = nt.inverse_minmax(qphys_predict.data.numpy(), qphys_scale.data.numpy(), qphys_feature_min.data.numpy(), qphys_feature_max.data.numpy(), qphys_data_min.data.numpy())
    qphys_predict_denorm = nt.inverse_std(qphys_predict.data.numpy(), nt.qphys_dot_stdscale_s.data.numpy(), nt.qphys_dot_mean_s.data.numpy())
    
    return qphys_predict_denorm.reshape(70)

def q_scm(region='50S69W'):
    """
    Using qphys prediction and then feeding them back into the emulator, 
    create a timeseries of prediction to compare with validation data
    """
    
    start = 0
    end = 1000
    q_ = nn_data.q_norm_test_s[start:end,:]
    q_raw =  nn_data.q_test_raw_s[start:end,:]
    qphys = nn_data.qphys_dot_norm_test_s[start:end,:]
    qadv = nn_data.qadv_norm_test[start:end,:]
    qadv_dot = nn_data.qadv_dot_norm_test_s[start:end,:]
    qadv_raw = nn_data.qadv_test_raw[start:end,:] # qadv_normaliser.inverse_transform(qadv)
    qadv_dot_raw = nn_data.qadv_dot_test_raw_s[start:end,:]
    # qadv_inv = nt.inverse_minmax(qadv, qadv_scale.data.numpy(), qadv_feature_min.data.numpy(), qadv_feature_max.data.numpy(), qadv_data_min.data.numpy()) 
    qadv_inv = nt.inverse_std(qadv, nt.qadv_stdscale.data.numpy(), nt.qadv_mean.data.numpy())
    # qadv_inv = nn_data.qadv_test_raw[start:end,:]

    # qadv_dot_inv = qadv_dot_raw * 600.
    qadv_dot_inv = qadv_dot_raw * 600.
 
    qcomb_test  = np.concatenate((nn_data.qadv_norm_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = np.concatenate((nn_data.qadd_dot_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = nn_data.qadd_dot_test[start:end,:]
    qcomb_dot_test  = np.concatenate((nn_data.q_norm_test_s, nn_data.qadv_dot_norm_test_s),axis=1)

    # qphys_inv = nt.inverse_minmax(qphys, qphys_scale.data.numpy(), qphys_feature_min.data.numpy(), qphys_feature_max.data.numpy(), qphys_data_min.data.numpy())
    # qphys_inv = nt.inverse_std(qphys, nt.qphys_dot_stdscale_s.data.numpy(), nt.qphys_dot_mean_s.data.numpy())
    qphys_inv = nn_data.qphys_dot_test_raw_s[start:end,:]

    n_steps = len(range(start,end))
    
    q_start = q_raw[0,:]
    q_ml = np.zeros((n_steps,70))
    q_sane = np.zeros((n_steps,70))
    qphys_drift = np.zeros((n_steps,70))
    qphys_pred_drift = np.zeros((n_steps,70))
    qphys_ml = np.zeros((n_steps,70))
    q_ml[0,:] = q_start
    q_sane[0,:] = q_start
    qphys_drift[0,:] = qphys[0,:]
    qphys_pred_drift[0,:] = qphys[0,:]
    qphys_ml[0,:] = qphys_inv[0,:]

    printProgressBar(0, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for t_step in range(1,n_steps):
        
        q_in = (qcomb_dot_test[t_step-1,:]).reshape((1,140))
        # q_in = (qcomb_dot_test[t_step,:]).reshape((1,70))
        qphys_pred = predict_q(q_in)
        qphys_ml[t_step,:] = qphys_pred
        #q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_inv[t_step,:] + qphys_pred
        # q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_pred
        q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_ml[t_step-1,:]
        q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_dot_inv[t_step-1,:] + qphys_inv[t_step-1, :]
        # q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_inv[t_step,:] + qphys_inv[t_step]
        qphys_drift[t_step,:] = qphys_drift[t_step-1,:] + qphys_inv[t_step]
        qphys_pred_drift[t_step,:] = qphys_pred_drift[t_step-1,:] + qphys_pred
        printProgressBar(t_step, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # qadv_un = qadv_inv #qadv_normaliser.inverse_transform(qadv_norm)
    output_dict = {"q_ml":q_ml[:],"q":q_raw[:],"qphys_ml":qphys_ml,"qphys":qphys_inv[:],"qadv":qadv_inv[:], "qadv_dot":qadv_dot_inv[:], "qadv_raw":qadv_raw[:], "qadv_dot_raw":qadv_dot_raw[:], "q_sane":q_sane[:], "qphys_drift":qphys_drift[:], "qphys_pred_drift":qphys_pred_drift[:]}
    outfile_name='scm_predict_{0}_epoch_{1}_qloss_0_qcomb_std.hdf5'.format('qloss',str(args.epochs).zfill(3))    
    with h5py.File(outfile_name,'w') as outfile:
        for k,v in output_dict.items():
            outfile.create_dataset(k,data=v)

def q_inference(region='50S69W'):
    """
    q model inference/testing
    """
    # Testing data
    qphys_norm_test = nn_data.qphys_dot_norm_test_s
    # qcomb_test  = np.concatenate((nn_data.qadv_norm_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = np.concatenate((nn_data.qadd_dot_test,nn_data.q_norm_test),axis=1)
    # qcomb_dot_test  = nn_data.qadd_dot_test
    qcomb_dot_test  = np.concatenate((nn_data.q_norm_test_s, nn_data.qadv_dot_norm_test_s),axis=1)
    #train_in, train_out = qcomb_dot_train, qphys_norm_train
    #test_in, test_out = qcomb_dot_test, qphys_norm_test
    x_t,y_t = torch.from_numpy(qcomb_dot_test[:]), qphys_norm_test[:]
    prediction = mlp(x_t)
    # qphys_predict_denorm = nt.inverse_minmax_tensor(prediction, qphys_scale, qphys_feature_min, qphys_feature_max, qphys_data_min)
    # qphys_test_denorm = nt.inverse_minmax_tensor(torch.from_numpy(qphys_norm_test[:]), qphys_scale, qphys_feature_min, qphys_feature_max, qphys_data_min)
    qphys_predict_denorm = nt.inverse_std(prediction, nt.qphys_dot_stdscale, nt.qphys_dot_mean)
    qphys_test_denorm = nt.inverse_std(torch.from_numpy(qphys_norm_test[:]), nt.qphys_dot_stdscale, nt.qphys_dot_mean)
    hfilename='qloss_qphys_predict_{0}deep_epoch_{1}_std.hdf5'.format(str(n_layers).zfill(3),str(args.epochs).zfill(3))
    output={'qphys_predict':qphys_predict_denorm.data.numpy(),'qphys_test':qphys_test_denorm.data.numpy()}
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
    # q_inference()
    q_scm()