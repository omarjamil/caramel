import tensorflow as tf
from tensorflow import keras
import pickle
import joblib
import numpy as np
import math
import sys
import os
import h5py

import mlp_model as mlm
import data_io

def inverse_data_transformation(scale, X_scaled, feature_min, X_min):
    """
    scale (n_features)
    X_scaled (n_samples, n_features)
    feature_min (n_features)
    X_min (n_features)
    """
    X = X_scaled/scale - feature_min + X_min
    return X

def data_transform(scale, X, feature_min, feature_max):
    """
    scale (n_features)
    X (n_samples, n_features)
    feature_min, feature_max = min and max for scaling
    """
    X_scaled = scale * X + feature_min - X.min(axis=0) * scale
    return X_scaled
    
# @tf.function
def q_loss(qphys_prediction, q_loss_data):
    """
    Extra loss for q predicted from the model
    """
    global qphys_normaliser
    global q_normaliser
    global qadd_normaliser
    qadd_dot = q_loss_data[0][:,:70]
    qnext_train = q_loss_data[1]
    qadd_dot_denorm = qadd_normaliser.inverse_transform(qadd_dot)
    qphys_prediction_denorm = qphys_normaliser.inverse_transform(qphys_prediction)
    qnext_calc = qadd_dot_denorm + qphys_prediction_denorm
    qnext_calc_norm = q_normaliser.transform(qnext_calc)

    loss = loss_object(qnext_calc_norm, qnext_train)
    return loss

# @tf.function
def train_step(inputs, outputs, q_next, epoch):
    """
    """
    global model
    # global loss_objects
    global optimizer
    global train_loss
    global train_accuracy
    extra_loss_data = [inputs,q_next]
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        qchange_loss = q_loss(predictions, extra_loss_data)
        loss = loss_object(outputs, predictions)
        total_loss = loss + qchange_loss
    # gradients = tape.gradient(loss, model.trainable_variables)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Modifying learning_rate during epochs
    #if epoch % 2 == 0:
    #    optimizer.learning_rate = 0.1
    #    tf.print(optimizer.learning_rate, output_stream=sys.stdout)

    # train_loss(loss)
    train_loss.update_state(outputs, predictions)
    train_accuracy.update_state(outputs, predictions)

# @tf.function
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

    q_next = train_data[2]
    train_in, train_out = train_data[0], train_data[1]
    test_in, test_out = test_data[0], test_data[1]

    train_ds = tf.data.Dataset.from_tensor_slices((train_in, train_out, q_next)).shuffle(10000).batch(BATCH)
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
        for inputs, outputs, q_next in train_ds:
            train_step(inputs, outputs, q_next, epoch)
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

def test_model(test_data: list, model_name: str='q', region: str="50S69W"):
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

qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
qadd_normaliser = joblib.load('{0}/minmax_qadd_dot.joblib'.format(locations['normaliser_loc']))

if __name__ == "__main__":
    BATCH=100
    EPOCHS=50
    # model_name = "qcomb_add_dot_qphys_deep"
    model_name = "qcomb_add_dot_qloss_qphys_deep"
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
    
    #train_in, train_out = qcomb_dot_train, qphys_norm_train
    #test_in, test_out = qcomb_dot_test, qphys_norm_test
    train_data = [qcomb_dot_train, qphys_norm_train, qnext_norm_train]
    test_data = [qcomb_dot_test, qphys_norm_test]
    
    # TRAIN the model
    create_train_model(BATCH, EPOCHS, train_data, test_data, model_name=model_name, region=region)
    
    # TEST the model
    qphys_predictions = test_model(test_data, model_name=model_name, region=region)
    qphys_normaliser = joblib.load('{0}/minmax_qphystot.joblib'.format(locations['normaliser_loc']))
    q_normaliser = joblib.load('{0}/minmax_qtot.joblib'.format(locations['normaliser_loc']))
    qphys_predict_denorm = qphys_normaliser.inverse_transform(qphys_predictions)
    qphys_test_denorm = qphys_normaliser.inverse_transform(qphys_norm_test)
    filename='{0}_{1}.hdf5'.format(model_name,region)
    testing_output={"qphys_predict":qphys_predict_denorm, "qphys_test":qphys_test_denorm}
    with h5py.File(filename, 'w') as hfile:
        for k, v in testing_output.items():  
            hfile.create_dataset(k,data=v)