import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

import mlp_model as mlm
import data_io
# Define various aspects of the model - global scope
# model = mlm.get_model_40_40()
# model = mlm.get_model_70_70()
# model = mlm.get_model_70_70__()
# model = mlm.get_model_140_70()
model = mlm.get_model_140_70_deep()

#model = mlm.get_model_280_140()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts",
            "pkl_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

def test_model_T(model_name='T', region="50S69W"):
    """
    Test the temperature model
    """
    global model
    # Load testing data
    train_test_datadir = "data/models/datain/"
    dataset=np.load(train_test_datadir+"train_test_data_{0}.npz".format(region))
    tadv_test = dataset['tadv_norm_test']
    t_test = dataset['t_norm_test']
    tphys_test = dataset['tphys_norm_test']
    tcomb_test = np.concatenate((tadv_test,t_test), axis=1)
        
    # checkpoint_dir = '/project/spice/radiation/ML/CRM/data/models/chkpts_gpu/{0}'.format(model_name)
    checkpoint_dir = 'data/models/chkpts/{0}'.format(model_name)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # After compiling model, run through some data for initialising variables then load weights
    model.compile(optimizer=optimizer,loss=loss_object,metrics=[test_accuracy])
    model(tcomb_test[:1])
    # model(t_test[:1])
    #model(tadv_test[:1])
    # model(qt_test[:1])
    model.load_weights(latest)
    print(model_name)
    print(model.summary())
    
    tphys_normaliser = joblib.load('data/models/normaliser/minmax_tphys.joblib')
    tphys_predict = model.predict(tcomb_test)
    # tphys_predict = model.predict(t_test)
    # tphys_predict = model.predict(tadv_test)
        
    tphys_predict_denorm = tphys_normaliser.inverse_transform(tphys_predict)
    tphys_test_denorm = tphys_normaliser.inverse_transform(tphys_test)

    # tphys_predict_denorm = tphys_predict
    # tphys_test_denorm = tphys_test

    np.savez('tcomb_predict_norm',tphys_predict=tphys_predict_denorm,tphys_test=tphys_test_denorm)

def test_model_q(model_name='q', region="50S69W"):
    """
    Test the humidity model
    """
    global model
    # Load testing data
    train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
    q_norm_test = test_data_in["qtot_test"]
    qadv_norm_test = test_data_in["qadv_test"]
    qadv_dot_norm_test = test_data_in["qadv_dot_test"]
    qphys_norm_test = test_data_out["qphys_test"]
    qadd_test = test_data_in["qadd_test"]
    qadd_dot_test = test_data_in["qadd_dot_test"]
    qcomb_test  = np.concatenate((qadv_norm_test,q_norm_test),axis=1)
    qcomb_dot_test  = np.concatenate((qadd_dot_test,q_norm_test),axis=1)
    # qcomb_dot_test  = np.concatenate((qadv_dot_norm_test,q_norm_test),axis=1)
    

    checkpoint_dir = '{0}/{1}'.format(locations['chkpnt_loc'], model_name)    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # After compiling model, run through some data for initialising variables then load weights
    model.compile(optimizer=optimizer,loss=loss_object,metrics=[test_accuracy])
    # model(qadv_test[:1])
    # model(q_test[:1])
    # model(qcomb_test[:1])
    model(qcomb_dot_test[:1])
    model.load_weights(latest)
    print(model_name)
    print(model.summary())
    
    qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
    q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
    qadv_normaliser = joblib.load('{0}/minmax_qadvtot.joblib'.format(locations['normaliser_loc']))
    qadv_dot_normaliser = joblib.load('{0}/minmax_qadv_dot.joblib'.format(locations['normaliser_loc']))
    # qphys_predict = model.predict(qadv_test)
    # qphys_predict = model.predict(q_test)
    # qphys_predict = model.predict(qcomb_test)
    qadv_predict = model.predict(qcomb_dot_test)
    
    # qphys_predict_denorm = qphys_normaliser.inverse_transform(qphys_predict)
    # qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_test)
    qadv_predict_denorm = qadv_normaliser.inverse_transform(qadv_predict)
    qadv_test_denorm = qadv_normaliser.inverse_transform(qadv_norm_test)
    # np.savez('qtot_predict',qphys_predict=qphys_predict_denorm,qphys_test=qphys_test_denorm)
    np.savez('qcomb_dot_predict_qadv',qadv_predict=qadv_predict_denorm,qadv_test=qadv_test_denorm)
        
def test_model(model_name='qT', region="50S69W"):
    global model
    
    ## dataset = di.driving_data()
    # q_train, q_test, qadv_train, qadv_test, qphys_train, qphys_test, t_train, t_test, tadv_train, tadv_test, tphys_train, tphys_test = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7]
    # Load testing data
    qt_train, qt_test, qt_phys_train, qt_phys_test = data_io.qt_model_data(region)
    qphys_test = qt_phys_test[:,:70]
    tphys_test = qt_phys_test[:,70:]

    # train_test_datadir = "data/models/datain/"
    # dataset=np.load(train_test_datadir+"train_test_data_{0}.npz".format(region))
    # qadv_test = dataset['qadv_norm_test']
    # q_test = dataset['q_norm_test']
    # qphys_test = dataset['qphys_norm_test']
    # qcomb_test  = np.concatenate((qadv_test,q_test),axis=1)
    # tadv_test = dataset['tadv_norm_test']
    # t_test = dataset['t_norm_test']
    # tphys_test = dataset['tphys_norm_test']
    # tcomb_test = np.concatenate((tadv_test,t_test), axis=1)
    # qt_test = np.concatenate((qcomb_test, tcomb_test), axis=1)
    
    checkpoint_dir = 'data/models/chkpts/{0}'.format(model_name)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    # After compiling model, run through some data for initialising variables then load weights
    model.compile(optimizer=optimizer,loss=loss_object,metrics=[test_accuracy])
    # model(qadv_test[:1])
    # model(q_test[:1])
    # model(qcomb_test[:1])
    # model(tcomb_test[:1])
    # model(t_test[:])
    # model(tadv_test[:1])
    model(qt_test[:1])
    model.load_weights(latest)
    print(model_name)
    print(model.summary())
    
    # qphys_normaliser = joblib.load('/project/spice/radiation/ML/CRM/data/models/normaliser/std_qphy.joblib')
    qphys_normaliser = joblib.load('data/models/normaliser/minmax_qphys_qcomb.joblib')
    tphys_normaliser = joblib.load('data/models/normaliser/minmax_tphys.joblib')
    # qphys_predict = model.predict(qadv_test)
    # qphys_predict = model.predict(q_test)
    # qphys_predict = model.predict(qcomb_test)
    # tphys_predict = model.predict(tcomb_test)
    # tphys_predict = model.predict(t_test)
    # tphys_predict = model.predict(tadv_test)
    qt_predict = model.predict(qt_test)
    

    qphys_predict_denorm = qphys_normaliser.inverse_transform(qt_predict[:,:70])
    tphys_predict_denorm = tphys_normaliser.inverse_transform(qt_predict[:,70:])
    qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_test)
    tphys_test_denorm = tphys_normaliser.inverse_transform(tphys_test)

    # qphys_predict_denorm = qt_predict[:,:70]
    # tphys_predict_denorm = qt_predict[:,70:]
    # qphys_test_denorm = qphys_test
    # tphys_test_denorm = tphys_test

    # np.savez('qadv_predict',qphys_predict=qphys_predict_denorm,qphys_test=qphys_test_denorm)
    # np.savez('tadv_predict',tphys_predict=tphys_predict_denorm,tphys_test=tphys_test_denorm)
    np.savez('qT_predict',qphys_predict=qphys_predict_denorm,qphys_test=qphys_test_denorm,tphys_predict=tphys_predict_denorm, tphys_test=tphys_test_denorm)
    
if __name__ == "__main__":
    BATCH=100
    EPOCHS=100
    # model_name = "qcomb_qphys"
    region = "50S69W"
    
    # test_model_T(model_name="tcomb_tphys", region=region)
    test_model_q(model_name="qcomb_dot_qadv_deep", region=region)
    # test_model(model_name="qT_elu", region=region)
