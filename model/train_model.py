import tensorflow as tf
from tensorflow import keras
import pickle
# import joblib
import numpy as np
import math
import sys
import os

import mlp_model as mlm
import data_io

# Define various aspects of the model - global scope
# model = mlm.get_model_40_40()
# model = mlm.get_model_70_70()
# model = mlm.get_model_140_70()
model = mlm.get_model_280_140()

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
            "pkl_loc":"/project/spice/radiation/ML/CRM/data/models/history"}

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

def create_train_model(BATCH, EPOCHS, model_name="qT", region="50S69W"):
    """
    """
    global model
    global train_loss
    global train_accuracy
    global test_loss
    global test_accuracy
    

    qt_train, qt_test, qt_phys_train, qt_phys_test = data_io.qt_model_data(region)
    # Q and T model
    train_ds = tf.data.Dataset.from_tensor_slices((qt_train, qt_phys_train)).shuffle(10000).batch(BATCH)
    test_ds = tf.data.Dataset.from_tensor_slices((qt_test, qt_phys_test)).batch(BATCH)
    
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


if __name__ == "__main__":
    BATCH=100
    EPOCHS=50
    model_name = "qT_elu"
    region="50S69W"
    create_train_model(BATCH, EPOCHS, model_name=model_name, region=region)
