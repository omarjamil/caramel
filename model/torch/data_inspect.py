import torch
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py
import argparse
import matplotlib.pyplot as plt

import model
import data_io
import normalize as nt

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

region="50S69W"

train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
q_norm_train = train_data_in["qtot"]
qnext_norm_train = train_data_in["qtot_next"]
qadv_norm_train = train_data_in["qadv"]
qadv_dot_norm_train = train_data_in["qadv_dot"]
qphys_norm_train = train_data_out["qphys_tot"]

q_norm_test = test_data_in["qtot_test"]
qnext_norm_test = test_data_in["qtot_next_test"]
qadv_norm_test = test_data_in["qadv_test"]
qadv_dot_norm_test = test_data_in["qadv_dot_test"]
qphys_norm_test = test_data_out["qphys_test"]
qadd_train = train_data_in["qadd"]
qadd_dot_train = train_data_in["qadd_dot"]
qadd_test = test_data_in["qadd_test"]
qadd_dot_test = test_data_in["qadd_dot_test"]
qcomb_train = np.concatenate((qadv_norm_train,q_norm_train),axis=1)
qcomb_test  = np.concatenate((qadv_norm_test,q_norm_test),axis=1)
# qcomb_dot_train = np.concatenate((qadv_dot_norm_train,q_norm_train),axis=1)
# qcomb_dot_test  = np.concatenate((qadv_dot_norm_test,q_norm_test),axis=1)
qcomb_dot_train = np.concatenate((qadd_dot_train,q_norm_train),axis=1)
qcomb_dot_test  = np.concatenate((qadd_dot_test,q_norm_test),axis=1)
# Data normalisers
qphys_normaliser = h5py.File('{0}/minmax_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
q_normaliser = h5py.File('{0}/minmax_qtot.hdf5'.format(locations['normaliser_loc']),'r')
qadd_normaliser = h5py.File('{0}/minmax_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
qphys_feature_min = torch.tensor(qphys_normaliser['feature_range'][0])
qphys_feature_max = torch.tensor(qphys_normaliser['feature_range'][1])
qadd_feature_min  = torch.tensor(qadd_normaliser['feature_range'][0])
q_feature_max = torch.tensor(q_normaliser['feature_range'][1])
q_feature_min  = torch.tensor(q_normaliser['feature_range'][0])
qadd_feature_max = torch.tensor(qadd_normaliser['feature_range'][1])
qphys_scale = torch.from_numpy(qphys_normaliser['scale_'][:])
qadd_scale = torch.from_numpy(qadd_normaliser['scale_'][:])
q_scale = torch.from_numpy(q_normaliser['scale_'][:])
qphys_data_min = torch.from_numpy(qphys_normaliser['data_min_'][:])
qadd_data_min = torch.from_numpy(qadd_normaliser['data_min_'][:])
q_data_min = torch.from_numpy(q_normaliser['data_min_'][:])
qadd_normaliser_sk = joblib.load('{0}/minmax_qadd_dot.joblib'.format(locations['normaliser_loc']))

def view_minmax_data():
    
    
    qadd_dot = qcomb_dot_train[:100,:70]
    qadd_dot_denorm = qadd_normaliser_sk.inverse_transform(qadd_dot)
    qadd_dot_denorm_tensor = nt.inverse_minmax(torch.from_numpy(qadd_dot), qadd_scale, qadd_feature_min, qadd_feature_max, qadd_data_min)
    qphys_denorm = nt.inverse_minmax(torch.from_numpy(qphys_norm_train[:]), qphys_scale, qphys_feature_min, qphys_feature_max, qphys_data_min)
    q_denorm = nt.inverse_minmax(torch.from_numpy(q_norm_train[:]), q_scale, q_feature_min, q_feature_max, q_data_min)

    #print(qadd_dot_denorm)
    #print(qadd_dot[:100,:70])
    #print(qadd_dot_denorm_tensor.data)
    qphys_norm_var = np.var(qphys_norm_train, axis=0)
    qphys_var = np.var(qphys_denorm.data.numpy(), axis=0)
    qphys_range = np.amax(qphys_denorm.data.numpy(), axis=0) - np.amin(qphys_denorm.data.numpy(),axis=0)
    qadd_norm_var = np.var(qadd_dot, axis=0)
    qadd_var = np.var(qadd_dot_denorm, axis=0)
    qadd_range = np.amax(qadd_dot_denorm, axis=0) - np.amin(qadd_dot_denorm,axis=0)
    q_norm_var = np.var(q_norm_train, axis=0)
    q_var = np.var(q_denorm.data.numpy(), axis=0)
    qadd_range = np.amax(q_denorm.data.numpy(), axis=0) - np.amin(q_denorm.data.numpy(), axis=0)

    fig, axs = plt.subplots(2,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.plot(qphys_norm_var, 'k-')
    c = ax.plot(qadd_norm_var,'r-')
    c = ax.plot(q_norm_var,'b-')

    ax = axs[1]
    c = ax.plot(qphys_var, 'k-')
    c = ax.plot(qadd_var,'r-')
    c = ax.plot(q_var,'b-')
    

    #c = ax.plot(qphys_range)
    plt.show()

if __name__ == "__main__":
    view_minmax_data()