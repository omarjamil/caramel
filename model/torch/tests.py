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
import normalize_tensors as nt

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

def test_tensor_normaliser():
    qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
    q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
    qadd_normaliser = joblib.load('{0}/minmax_qadd_dot.joblib'.format(locations['normaliser_loc']))
    qphys_normaliser_tensor = h5py.File('{0}/minmax_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
    q_normaliser_tensor = h5py.File('{0}/minmax_qtot.hdf5'.format(locations['normaliser_loc']),'r')
    qadd_normaliser_tensor = h5py.File('{0}/minmax_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
    
    qadd_dot = qcomb_dot_train[:100,:70]
    qadd_dot_denorm = qadd_normaliser.inverse_transform(qadd_dot)
    qadd_dot_denorm_tensor = nt.inverse_minmax_tensor(qadd_normaliser_tensor, torch.from_numpy(qadd_dot))
    #print(qadd_dot_denorm)
    #print(qadd_dot[:100,:70])
    #print(qadd_dot_denorm_tensor.data)
    print(qadd_dot_denorm_tensor.data.numpy() -  qadd_dot_denorm)
if __name__ == "__main__":
    test_tensor_normaliser()
