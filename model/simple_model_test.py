import tensorflow as tf
from tensorflow import keras
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt

import data_input as di
import MLModel as mlm



# qadv_train, qadv_test, qphys_train, qphys_test,  tadv_train, tadv_test, tphys_train, tphys_test


def create_and_train():
    
    #q_train, q_test, qadv_train, qadv_test, qphys_train, qphys_test, t_train, t_test, tadv_train, tadv_test, tphys_train, tphys_test = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7], dataset[8], dataset[9], dataset[10], dataset[11]
    train_test_datadir = "/project/spice/radiation/ML/CRM/data/models/"
    dataset=np.load(train_test_datadir+'train_test_data.npz')
    qadv_train = dataset['qadv_train']
    q_train = dataset['q_train']
    qphys_train = dataset['qphys_train']
    qphys_test = dataset['qphys_test']
    q_test = dataset['q_test']
    # model = tf.keras.Sequential([
    #     keras.layers.Dense(128, activation='relu', input_shape=(70,)),
    #     keras.layers.Dense(256, activation='relu'),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(70, activation='tanh')
    #     ])
    model = mlm.get_model()
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,loss=loss,metrics=[tf.keras.metrics.Accuracy()])
    batch=100
    n_epoch=20
    # history = model.fit(qadv_train, qphys_train,epochs=n_epoch,batch_size=batch, validation_data=(qadv_test,qphys_test))
    history = model.fit(q_train, qphys_train,epochs=n_epoch,batch_size=batch, validation_data=(q_test,qphys_test)) 
    model_fname="model_q_epoch_{0}".format(n_epoch)
    model.save('/project/spice/radiation/ML/CRM/data/models/'+model_fname+'.h5')
    model.save_weights('/project/spice/radiation/ML/CRM/data/models/chkpts_keras/'+model_fname)
    pickle_file = '/project/spice/radiation/ML/CRM/data/models/'+model_fname+'.history'
    pickle.dump(history.history,open(pickle_file,'wb'))
    dataset.close()
    
def test_model():

    # dataset = di.driving_data()
    # q_train, q_test, qadv_train, qadv_test, qphys_train, qphys_test, t_train, t_test, tadv_train, tadv_test, tphys_train, tphys_test = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7]
    # Load testing data
    train_test_datadir = "/project/spice/radiation/ML/CRM/data/models/"
    dataset=np.load(train_test_datadir+'train_test_data.npz')
    qadv_test = dataset['qadv_test']
    q_test = dataset['q_test']
    qphys_test = dataset['qphys_test']
    
    model = tf.keras.models.load_model('/project/spice/radiation/ML/CRM/data/models/model_q_epoch_20.h5')
    #qphys_normaliser = joblib.load('/project/spice/radiation/ML/CRM/data/models/normaliser/std_qphy.joblib')
    qphys_normaliser = joblib.load('/project/spice/radiation/ML/CRM/data/models/normaliser/minmax_qphys.joblib')
    qphys_predict = model.predict(q_test)
    qphys_predict_denorm = qphys_normaliser.inverse_transform(qphys_predict)
    qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_test)
    np.savez('qqphys_predict',qphys_predict=qphys_predict_denorm,qphys_test=qphys_test_denorm,qphys_test_norm=qphys_test)
    # print(qphys_predict_denorm.shape, qphys_test_denorm.shape)

def test_model_weights():

    ## dataset = di.driving_data()
    # q_train, q_test, qadv_train, qadv_test, qphys_train, qphys_test, t_train, t_test, tadv_train, tadv_test, tphys_train, tphys_test = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7]
    # Load testing data
    train_test_datadir = "/project/spice/radiation/ML/CRM/data/models/"
    dataset=np.load(train_test_datadir+'train_test_data.npz')
    qadv_test = dataset['qadv_test']
    q_test = dataset['q_test']
    qphys_test = dataset['qphys_test']

    checkpoint_dir = '/project/spice/radiation/ML/CRM/data/models/chkpts_keras/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # model = tf.keras.Sequential([
    #     keras.layers.Dense(128, activation='relu', input_shape=(70,)),
    #     keras.layers.Dense(256, activation='relu'),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(70, activation='tanh')
    #     ])
    model = mlm.get_model()

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,loss=loss,metrics=[tf.keras.metrics.Accuracy()])
    model.load_weights(latest)
    #qphys_normaliser = joblib.load('/project/spice/radiation/ML/CRM/data/models/normaliser/std_qphy.joblib')
    qphys_normaliser = joblib.load('/project/spice/radiation/ML/CRM/data/models/normaliser/minmax_qphys.joblib')
    qphys_predict = model.predict(q_test)
    qphys_predict_denorm = qphys_normaliser.inverse_transform(qphys_predict)
    qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_test)
    np.savez('qqphys_predict_wts',qphys_predict=qphys_predict_denorm,qphys_test=qphys_test_denorm,qphys_test_norm=qphys_test)

def visualise_predictions():
    data = np.load('qphys_predict_manual.npz')
    qphys_predict = data['qphys_predict'].T
    qphys_test = data['qphys_test'].T
    qphys_test_norm = data['qphys_test_norm'].T
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.pcolor(qphys_predict[:,0:5000])
    ax.set_title('Q predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(qphys_test[:,0:5000])
    ax.set_title('Q Test')
    fig.colorbar(c,ax=ax)

    diff = qphys_predict - qphys_test
    ax = axs[2]
    c = ax.pcolor(diff[:,0:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    
    figname="qadv_predict_manual.png"
    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()
    
if __name__ == "__main__":
    #create_and_train()
    #test_model()
    #test_model_weights()
    visualise_predictions()
