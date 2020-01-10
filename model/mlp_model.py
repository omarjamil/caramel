import tensorflow as tf
from tensorflow import keras
# Multi-layer Perceptron Model class
# 70 inputs, 70 outputs
class MLPModel_70_70(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_70_70, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(128, activation='relu', input_shape=(70,))
        self.dense_2 = keras.layers.Dense(256, activation='relu')
        self.dense_3 = keras.layers.Dense(128, activation='relu')
        self.pred_layer = keras.layers.Dense(70, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.pred_layer(x)

class MLPModel_70_70_(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_70_70_, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(128, activation='relu', input_shape=(70,), kernel_regularizer=keras.regularizers.l2(l=0.1))
        self.dense_2 = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1))
        self.dense_3 = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1))
        self.pred_layer = keras.layers.Dense(70, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.pred_layer(x)
    
class MLPModel_70_70__(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_70_70__, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(128, activation='selu', input_shape=(70,))
        self.dense_2 = keras.layers.Dense(256, activation='selu')
        self.dense_3 = keras.layers.Dense(128, activation='selu')
        self.pred_layer = keras.layers.Dense(70, activation='linear')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.pred_layer(x)

class MLPModel_40_40(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_40_40, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(128, activation='relu', input_shape=(40,))
        self.dense_2 = keras.layers.Dense(256, activation='relu')
        self.dense_3 = keras.layers.Dense(128, activation='relu')
        self.pred_layer = keras.layers.Dense(40, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.pred_layer(x)
    
# 140 inputs, 70 outputs
class MLPModel_140_70(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_140_70, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(256, activation='relu', input_shape=(140,))
        self.dense_2 = keras.layers.Dense(512, activation='relu')
        self.dense_3 = keras.layers.Dense(256, activation='relu')
        self.dense_4 = keras.layers.Dense(128, activation='relu')
        self.pred_layer = keras.layers.Dense(70, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.pred_layer(x)

# 140 inputs, 70 outputs
class MLPModel_140_70_deep(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_140_70_deep, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(256, activation='relu', input_shape=(140,))
        self.dense_2 = keras.layers.Dense(512, activation='relu')
        self.dense_3 = keras.layers.Dense(512, activation='relu')
        self.dense_4 = keras.layers.Dense(512, activation='relu')
        self.dense_5 = keras.layers.Dense(512, activation='relu')
        self.dense_6 = keras.layers.Dense(128, activation='relu')
        self.pred_layer = keras.layers.Dense(70, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        return self.pred_layer(x)


# 280 inputs, 140 outputs
class MLPModel_280_140(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_280_140, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(512, activation='elu', input_shape=(280,))
        self.dense_2 = keras.layers.Dense(768, activation='elu')
        self.dense_3 = keras.layers.Dense(1024, activation='elu')
        self.dense_4 = keras.layers.Dense(768, activation='elu')
        self.dense_5 = keras.layers.Dense(512, activation='elu')
        self.dense_6 = keras.layers.Dense(256, activation='elu')
        self.pred_layer = keras.layers.Dense(140, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        return self.pred_layer(x)

class MLPModel_284_146(tf.keras.Model):
    def __init__(self, name=None):
        super(MLPModel_284_146, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(512, activation='elu', input_shape=(284,))
        self.dense_2 = keras.layers.Dense(512, activation='elu')
        self.dense_3 = keras.layers.Dense(512, activation='elu')
        self.dense_4 = keras.layers.Dense(512, activation='elu')
        self.dense_5 = keras.layers.Dense(512, activation='elu')
        self.dense_6 = keras.layers.Dense(512, activation='elu')
        self.dense_7 = keras.layers.Dense(512, activation='elu')
        self.dense_8 = keras.layers.Dense(512, activation='elu')
        self.pred_layer = keras.layers.Dense(146, activation='tanh')
    
    def call(self,inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        x = self.dense_7(x)
        x = self.dense_8(x)
        return self.pred_layer(x)

def get_model_40_40():
    return MLPModel_40_40(name='MLPModel_40_40')

def get_model_70_70():
    return MLPModel_70_70(name='MLPModel_70_70')

def get_model_70_70__():
    return MLPModel_70_70__(name='MLPModel_70_70__')

def get_model_70_70_():
    return MLPModel_70_70_(name='MLPModel_70_70_')
    
def get_model_140_70():
    return MLPModel_140_70(name='MLPModel_140_70')

def get_model_140_70_deep():
    return MLPModel_140_70_deep(name='MLPModel_140_70_deep')
    
def get_model_280_140():
    return MLPModel_280_140(name='MLPModel_280_140')    

def get_model_284_146():
    return MLPModel_284_146(name='MLPModel_284_146')  
# def get_model():
#     inputs = keras.layers.Input(shape=(70,))
#     x = keras.layers.Dense(128, activation='relu')(inputs)
#     x = keras.layers.Dense(256, activation='relu')(x)
#     x = keras.layers.Dense(128, activation='relu')(x)
#     outputs = keras.layers.Dense(70, activation='tanh')(x)
#     return keras.Model(inputs=inputs, outputs=outputs)
