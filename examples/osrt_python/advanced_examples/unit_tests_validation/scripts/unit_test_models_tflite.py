import tensorflow as tf
import numpy as np
from tensorflow import keras
 
inputs = tf.keras.layers.Input(shape=[128, 128, 3])
input2 = tf.keras.layers.Input(shape=[128, 128, 3])
input3 = tf.keras.layers.Input(shape=[11, 1, 128])
input4 = tf.keras.layers.Input(shape=[1, 1, 3])
input5 = tf.keras.layers.Input(shape=[3,])

models_path = "../unit_test_models/"

input_vec = tf.constant(np.ones(3), np.float32)
input_tensor = tf.constant(np.random.rand(1, 1, 3), np.float32)
#------------------------------ Add test cases ------------------------------------------
#----------------------------------------------------------------------------------------
#element wise add
x = tf.keras.layers.Add()
y = x([inputs, input2])

model = tf.keras.Model(inputs=[inputs,input2], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "add_eltwise.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
# add constant to layer
y = inputs + tf.constant(2.)
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "add_const.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
# add constant vector to layer
x = tf.keras.layers.Add()
#y = x([inputs, input_tensor])
y = inputs + input_tensor
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "add_const_vec.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------

# add noneltwise
x = tf.keras.layers.Add()
y = x([inputs, input4])
model = tf.keras.Model(inputs=[inputs, input4], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "add_noneltwise.tflite", "wb").write(tflite_model)

#------------------------------------- Multiplication test cases --------------------------------------
#----------------------------------------------------------------------------------------
#element wise mul
x = tf.keras.layers.Multiply()
y = x([inputs, input2])

model = tf.keras.Model(inputs=[inputs,input2], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "mul_eltwise.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
# mul constant to layer
y = inputs * tf.constant(2.)
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "mul_const.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
# mul constant vector to layer
y = inputs * input_tensor
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "mul_const_vec.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------

# mul noneltwise
x = tf.keras.layers.Multiply()
y = x([inputs, input4])
model = tf.keras.Model(inputs=[inputs, input4], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "mul_noneltwise.tflite", "wb").write(tflite_model)

#------------------------- Fully connected ----------------------------------------------
#----------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((16,16,3)),
    keras.layers.AveragePooling2D(pool_size=(16,16)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, use_bias=True, bias_initializer=tf.random_uniform_initializer())
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "fully_connected.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((16,16,3)),
    keras.layers.AveragePooling2D(pool_size=(16,16)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "fully_connected_no_bias.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((16,16,3)),
    keras.layers.AveragePooling2D(pool_size=(16,16)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, use_bias=True, bias_initializer=tf.random_uniform_initializer(), activation='relu')
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "fully_connected_relu.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------

model = keras.Sequential([
    keras.layers.InputLayer((3,)),
    keras.layers.Dense(10)
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "fully_connected_trial.tflite", "wb").write(tflite_model)

#------------------------------------------------------------------------
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "fully_connected_TIDL_1816.tflite", "wb").write(tflite_model)


x = tf.keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='valid',
                kernel_initializer=tf.random_uniform_initializer(),
                bias_initializer=tf.random_uniform_initializer())
y = tf.nn.leaky_relu(x(inputs), alpha=0.2)
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "leakyRelu.tflite", "wb").write(tflite_model)

#--------------------------Conv transpose------------------------------------------
#----------------------------------------------------------------------------------

model = keras.Sequential([
    keras.layers.InputLayer((128,128,3)),
    keras.layers.Conv2DTranspose(16, (4,4), strides=(2, 2), padding='same',
                kernel_initializer=tf.random_uniform_initializer(),
                bias_initializer=tf.random_uniform_initializer())
])
#model.input.set_shape(1 + model.input.shape[1:])
input_name = model.input_names[0]
index = model.input_names.index(input_name)
model.inputs[index].set_shape([1, 128, 128,3])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "deconv2d_4x4_s2_bias.tflite", "wb").write(tflite_model)

#-------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((128,128,3)),
    keras.layers.Conv2DTranspose(16, (4,4), strides=(2, 2), padding='same',
                kernel_initializer=tf.random_uniform_initializer())
])
#model.input.set_shape(1 + model.input.shape[1:])
input_name = model.input_names[0]
index = model.input_names.index(input_name)
model.inputs[index].set_shape([1, 128, 128,3])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "deconv2d_4x4_s2_nobias.tflite", "wb").write(tflite_model)

#-------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((128,128,3)),
    keras.layers.Conv2DTranspose(16, (3,3), strides=(2, 2), padding='same',
                kernel_initializer=tf.random_uniform_initializer())
])
#model.input.set_shape(1 + model.input.shape[1:])
input_name = model.input_names[0]
index = model.input_names.index(input_name)
model.inputs[index].set_shape([1, 128, 128,3])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "deconv2d_3x3_s2.tflite", "wb").write(tflite_model)

#-------------------------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.InputLayer((128,128,3)),
    keras.layers.Conv2DTranspose(16, (2,2), strides=(2, 2), padding='same',
                kernel_initializer=tf.random_uniform_initializer())
])
#model.input.set_shape(1 + model.input.shape[1:])
input_name = model.input_names[0]
index = model.input_names.index(input_name)
model.inputs[index].set_shape([1, 128, 128,3])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "deconv2d_2x2_s2.tflite", "wb").write(tflite_model)

#---------------------------------- Division test case -----------------------------
#-----------------------------------------------------------------------------------
# div constant to layer
model = keras.Sequential([
    keras.layers.InputLayer((128,128,3)),
    keras.layers.Lambda(lambda x : x / 2.0)
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "div_const.tflite", "wb").write(tflite_model)

#----------------------------------------------------------------------------------------
# div constant vector to layer
y = inputs / input_tensor
model = tf.keras.Model(inputs=[inputs], outputs=y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(models_path + "div_const_vec.tflite", "wb").write(tflite_model)