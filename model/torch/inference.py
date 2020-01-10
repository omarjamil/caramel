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

n_inputs,n_outputs=140,70
mlp = model.MLP(n_inputs,n_outputs)
optimizer =  torch.optim.Adam(mlp.parameters())
loss_function = torch.nn.MSELoss()

model_fname = 'qcomb_add_dot_qloss_qphys_deep.tar'
checkpoint = torch.load(locations['model_loc']+'/'+model_fname)
mlp.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']
epoch = checkpoint['epoch']
mlp.eval() 
# or to continue traininig
# mlp.train()

def q_inference(region='50S69W'):
    """
    q model inference/testing
    """
    train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
    # qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
    qphys_normaliser = h5py.File('{0}/minmax_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
    q_normaliser = h5py.File('{0}/minmax_qtot.hdf5'.format(locations['normaliser_loc']),'r')
    qadd_normaliser = h5py.File('{0}/minmax_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
    
    q_norm_test = test_data_in["qtot_test"]
    qnext_norm_test = test_data_in["qtot_next_test"]
    qadv_norm_test = test_data_in["qadv_test"]
    qadv_dot_norm_test = test_data_in["qadv_dot_test"]
    qphys_norm_test = test_data_out["qphys_test"]
    qadd_test = test_data_in["qadd_test"]
    qadd_dot_test = test_data_in["qadd_dot_test"]
    qcomb_test  = np.concatenate((qadv_norm_test,q_norm_test),axis=1)
    # qcomb_dot_test  = np.concatenate((qadv_dot_norm_test,q_norm_test),axis=1)
    qcomb_dot_test  = np.concatenate((qadd_dot_test,q_norm_test),axis=1)

    #train_in, train_out = qcomb_dot_train, qphys_norm_train
    #test_in, test_out = qcomb_dot_test, qphys_norm_test
    x_t,y_t = torch.from_numpy(qcomb_dot_test), qphys_norm_test
    prediction = mlp(x_t)
    # qphys_predict_denorm = qphys_normaliser.inverse_transform(prediction.data.numpy())
    # qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_norm_test)
    qphys_predict_denorm = nt.inverse_minmax_tensor(qphys_normaliser, prediction)
    qphys_test_denorm = nt.inverse_minmax_tensor(qphys_normaliser, torch.from_numpy(qphys_norm_test[:]))
    
    hfilename='qcomb_add_dot_qloss_predict.hdf5'
    output={'qphys_predict':qphys_predict_denorm.data.numpy(),'qphys_test':qphys_test_denorm.data.numpy()}
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)

if __name__ == "__main__":
   q_inference() 