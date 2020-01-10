import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
import h5py

import mlp_model as mlm
import data_io


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
        
def predict_q(q_in):
    """
    Test the humidity model
    """
    global model
    global qphys_normaliser
    
    qphys_predict = model.predict(q_in)
        
    qphys_predict_denorm = qphys_normaliser.inverse_transform(qphys_predict)
    return qphys_predict_denorm.reshape(70)

def q_run():
    """
    Using qphys prediction and then feeding them back into the emulator, 
    create a timeseries of prediction to compare with validation data
    """
    # global qadv_normaliser

    n_steps = len(q_[:1000,0])
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
    printProgressBar(0, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for t_step in range(1,n_steps):
        # q_in = (q_[t_step,:]).reshape((1,70))
        # q_in = (qadv[t_step,:]).reshape((1,70))
        # q_in = (qadv_dot[t_step,:]).reshape((1,70))
        # q_in = (qcomb_test[t_step,:]).reshape((1,140))
        q_in = (qcomb_dot_test[t_step,:]).reshape((1,140))
        # q_in = (qadd_test[t_step,:]).reshape((1,70))
        # q_in = (qadd_dot_test[t_step,:]).reshape((1,70))
        qphys_pred = predict_q(q_in)
        qphys_ml[t_step,:] = qphys_pred
        q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_inv[t_step,:] + qphys_pred
        # q_ml[t_step,:] = q_ml[t_step-1,:] + qadv_dot_inv[t_step,:] + qphys_pred
        # q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_dot_inv[t_step,:] + qphys_inv[t_step]
        q_sane[t_step,:] = q_sane[t_step-1,:] + qadv_inv[t_step,:] + qphys_inv[t_step]
        qphys_drift[t_step,:] = qphys_drift[t_step-1,:] + qphys_inv[t_step]
        qphys_pred_drift[t_step,:] = qphys_pred_drift[t_step-1,:] + qphys_pred
        printProgressBar(t_step, n_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # qadv_un = qadv_inv #qadv_normaliser.inverse_transform(qadv_norm)
    output_dict = {"q_ml":q_ml,"q":q_raw[:n_steps,:],"qphys_ml":qphys_ml,"qphys":qphys_inv[:n_steps,:],"qadv":qadv_inv[:n_steps,:], "qadv_dot":qadv_dot_inv[:n_steps,:], "qadv_raw":qadv_raw[:n_steps], "qadv_dot_raw":qadv_dot_raw[:n_steps], "q_sane":q_sane, "qphys_drift":qphys_drift, "qphys_pred_drift":qphys_pred_drift}
    outfile_name='scm_predict_{0}.hdf5'.format(model_name)    
    with h5py.File(outfile_name,'w') as outfile:
        for k,v in output_dict.items():
            outfile.create_dataset(k,data=v)


# Global variables       
# Define various aspects of the model - global scope
# model = mlm.get_model_40_40()
# model = mlm.get_model_70_70()
# model = mlm.get_model_70_70__()
# model = mlm.get_model_140_70()
model = mlm.get_model_140_70_deep()
# model = mlm.get_model_280_140()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
# model_name='qcomb_add_dot_qphys_deep'
model_name = 'qcomb_add_dot_qloss_qphys_deep'
region="50S69W"

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts",
            "pkl_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

checkpoint_dir = '{0}/{1}'.format(locations['chkpnt_loc'], model_name)
latest = tf.train.latest_checkpoint(checkpoint_dir)
# After compiling model, run through some data for initialising variables then load weights
model.compile(optimizer=optimizer,loss=loss_object,metrics=[test_accuracy])

qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
qadv_normaliser = joblib.load('{0}/minmax_qadvtot.joblib'.format(locations['normaliser_loc']))
qadv_dot_normaliser = joblib.load('{0}/minmax_qadv_dot.joblib'.format(locations['normaliser_loc']))
# qadv_norm_train, qadv_norm_test, qadv_train, qadv_test, q_norm_train, q_norm_test, q_train, q_test, qphys_norm_train, qphys_norm_test, qphys_train, qphys_test = data_io.q_scm_data(region)
train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
q_ = test_data_in["qtot_test"]
q_raw = test_data_in["qtot_test_raw"]
qphys = test_data_out["qphys_test"]
qadv = test_data_in["qadv_test"]
qadv_dot = test_data_in["qadv_dot_test"]
qadv_raw = test_data_in["qadv_test_raw"] # qadv_normaliser.inverse_transform(qadv)
qadv_dot_raw = test_data_in["qadv_dot_test_raw"]
qadv_inv = qadv_normaliser.inverse_transform(qadv)
#qadv_dot_inv = qadv_dot_normaliser.inverse_transform(qadv_dot)*600. #It's units are kg/kg/s so to get increment over 10 minutes x600
qadv_dot_inv = qadv_dot_raw[:] * 600.
qadd_test = test_data_in["qadd_test"]
qadd_dot_test = test_data_in["qadd_dot_test"]
qcomb_test  = np.concatenate((qadv, q_),axis=1)
# qcomb_dot_test  = np.concatenate((qadv_dot, q_),axis=1)
qcomb_dot_test  = np.concatenate((qadd_dot_test,q_),axis=1)
qphys_inv = qphys_normaliser.inverse_transform(qphys) # test_data_out["qphys_test_raw"] # 


# model(qadv[:1])
# model(qadv_dot[:1])
# model(q_test[:1])
# model(q_[:1])
# model(qcomb_test[:1])
model(qcomb_dot_test[:1])
# model(qadd_test[:1])
# model(qadd_dot_test[:1])
model.load_weights(latest)
print(model_name)
print(model.summary())

if __name__ == "__main__":
    q_run()
        
