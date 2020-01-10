import tensorflow as tf
from tensorflow import keras
import pickle
import joblib
import numpy as np
import math
import sys
import os
import joblib
import h5py

import mlp_model as mlm
import data_io

@tf.function
def train_step(inputs, outputs, epoch):
    """
    """
    global model
    # global loss_objects
    global optimizer
    global train_loss
    global train_accuracy
    
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Modifying learning_rate during epochs
    #if epoch % 2 == 0:
    #    optimizer.learning_rate = 0.1
    #    tf.print(optimizer.learning_rate, output_stream=sys.stdout)

    # train_loss(loss)
    train_loss.update_state(outputs, predictions)
    train_accuracy.update_state(outputs, predictions)

@tf.function
def test_step(inputs, outputs):
    """
    """
    global model
    # global loss_objects
    global test_loss
    global test_accuracy
    
    predictions = model(inputs)
    # t_loss = loss_object(outputs, predictions)
    # test_loss(t_loss)
    
    test_loss.update_state(outputs, predictions)
    test_accuracy.update_state(outputs, predictions)

def create_train_model(BATCH: int, EPOCHS: int, train_data: list, test_data: list, model_name: str="q", region: str="50S69W"):
    """
    """
    global model
    global train_loss
    global train_accuracy
    global test_loss
    global test_accuracy

    train_in, train_out = train_data[0], train_data[1]
    test_in, test_out = test_data[0], test_data[1]

    train_ds = tf.data.Dataset.from_tensor_slices((train_in, train_out)).shuffle(10000).batch(BATCH)
    test_ds = tf.data.Dataset.from_tensor_slices((test_in, test_out)).batch(BATCH)            
    train_loss_results = []
    train_accuracy_results = []
    
    model_fname="model_{0}_epochs_{1}".format(model_name, EPOCHS)
    checkpoint_dir = "{0}/{1}/".format(locations["chkpnt_loc"],model_name)
    try:
        os.makedirs(checkpoint_dir)
    except OSError:
        pass
    print(model_name, model.name)
    for epoch in range(EPOCHS):
        for inputs, outputs in train_ds:
            train_step(inputs, outputs, epoch)
            # print("Epoch: {0} Loss: {1} Accuracy: {2}".format(epoch+1,train_loss.result(),train_accuracy.result()*100), end='\r')
        
        for test_inputs, test_outputs in test_ds:
            test_step(test_inputs, test_outputs)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))
        
        train_loss_results.append(train_loss.result())
        train_accuracy_results.append(train_accuracy.result())
        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # Intermediate checkpoint saving
        checkpoint_name = "model_{0}_epoch_{1}".format(model_name, str(epoch).zfill(4))
        if epoch%10 == 0:
            model.save_weights(checkpoint_dir+checkpoint_name)
    # Save the model in checkpoints
    # https://www.tensorflow.org/guide/keras/save_and_serialize#saving_subclassed_models
    
    # Final checkpoint    
    model.save_weights(checkpoint_dir+model_fname)
    # Save model training history
    history = {'training_loss':train_loss_results,
               'training_accuracy':train_accuracy_results}
    pickle_file = "{0}/{1}.history".format(locations["pkl_loc"],model_fname)
    pickle.dump(history,open(pickle_file,'wb'))

def test_model(test_data: list, model_name: str='t', region: str="50S69W"):
    """
    Test the humidity model
    """
    global model
    
    checkpoint_dir = '{0}/{1}'.format(locations['chkpnt_loc'], model_name)    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # After compiling model, run through some data for initialising variables then load weights
    model.compile(optimizer=optimizer,loss=loss_object,metrics=[test_accuracy])
    
    model(test_data[0][:1])
    model.load_weights(latest)
    print(model_name)
    print(model.summary())
    
    predict = model.predict(test_data[0])
       
    return predict

#Global Variables
model = mlm.get_model_140_70_deep()
# model = mlm.get_model_140_70_deep()
# model = mlm.get_model_280_140()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.SGD(lr=0.1)
# tf.print(optimizer.learning_rate, output_stream=sys.stdout)
train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
            "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts",
            "pkl_loc":"/project/spice/radiation/ML/CRM/data/models/history",
            "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser"}

if __name__ == "__main__":
    BATCH=100
    EPOCHS=50
    # model_name = "qcomb_add_dot_qphys_deep"
    model_name = "tcomb_add_dot_qphys_deep"
    region="50S69W"

    train_data_in, train_data_out, test_data_in, test_data_out = data_io.scm_model_data(region)
    t_norm_train = train_data_in["T"]
    tadv_norm_train = train_data_in["tadv"]
    tadv_dot_norm_train = train_data_in["tadv_dot"]
    tphys_norm_train = train_data_out["tphys_tot"]
    
    t_norm_test = test_data_in["T_test"]
    tadv_norm_test = test_data_in["tadv_test"]
    tadv_dot_norm_test = test_data_in["tadv_dot_test"]
    tphys_norm_test = test_data_out["tphys_test"]
    tadd_train = train_data_in["tadd"]
    tadd_dot_train = train_data_in["tadd_dot"]
    tadd_test = test_data_in["tadd_test"]
    tadd_dot_test = test_data_in["tadd_dot_test"]
    tcomb_train = np.concatenate((tadv_norm_train,t_norm_train),axis=1)
    tcomb_test  = np.concatenate((tadv_norm_test,t_norm_test),axis=1)
    # qcomb_dot_train = np.concatenate((qadv_dot_norm_train,q_norm_train),axis=1)
    # qcomb_dot_test  = np.concatenate((qadv_dot_norm_test,q_norm_test),axis=1)
    tcomb_dot_train = np.concatenate((tadd_dot_train,t_norm_train),axis=1)
    tcomb_dot_test  = np.concatenate((tadd_dot_test,t_norm_test),axis=1)
    
    #train_in, train_out = qcomb_dot_train, qphys_norm_train
    #test_in, test_out = qcomb_dot_test, qphys_norm_test
    train_data = [tcomb_dot_train, tphys_norm_train]
    test_data = [tcomb_dot_test, tphys_norm_test]
    
    # TRAIN the model
    create_train_model(BATCH, EPOCHS, train_data, test_data, model_name=model_name, region=region)
    
    # TEST the model
    tphys_predictions = test_model(test_data, model_name=model_name, region=region)
    tphys_normaliser = joblib.load('{0}/minmax_tphys.joblib'.format(locations['normaliser_loc']))
    t_normaliser = joblib.load('{0}/minmax_T.joblib'.format(locations['normaliser_loc']))
    tphys_predict_denorm = tphys_normaliser.inverse_transform(tphys_predictions)
    tphys_test_denorm = tphys_normaliser.inverse_transform(tphys_norm_test)
    filename='{0}_{1}.hdf5'.format(model_name,region)
    testing_output={"tphys_predict":tphys_predict_denorm, "tphys_test":tphys_test_denorm}
    with h5py.File(filename, 'w') as hfile:
        for k, v in testing_output.items():  
            hfile.create_dataset(k,data=v)