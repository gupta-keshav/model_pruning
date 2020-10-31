# importing the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(48)

# I will use the MNIST dataset for this task, loading it from the tensorflow itself
dataset = tf.keras.datasets.mnist.load_data()
X_train, y_train = dataset[0]
X_test, y_test = dataset[1]
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_train.shape

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from kerassurgeon.operations import delete_channels, Surgeon

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu', name='conv_1'))
model.add(Conv2D(32, (3, 3), activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv_3'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv_4'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, to_categorical(y_train, 10), batch_size=32, epochs=5)

print(model.evaluate(X_test, to_categorical(y_test)))

weights_conv = model.get_layer('conv_3').get_weights()[0] #getting the weights of the layer

weights_dict = {}
num_filters = len(weights_conv[0, 0, 0, :])
for j in range(num_filters):
    w_s = np.sum(abs(weights_conv[:, :, :, j])) # l1_norm of the channel j
    filt = f'filt_{j}'
    weights_dict[filt] = w_s 

weights_dict_sort = sorted(weights_dict.items(), key=lambda kv: kv[1]) #dictionary containing the filter number and its l1_norm sorted in ascending order according to the norm
print(weights_dict_sort)

num_channels = 8 #number of channels to be deleted
layer_3 = model.get_layer('conv_3') #layer from which the channels are to be deleted
channels_3 = [int(weights_dict_sort[i][0].split('_')[1]) for i in range(num_channels)]
model_new = delete_channels(model, layer_3, channels_3)
model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_new.evaluate(X_test, to_categorical(y_test)))

weights_conv = model.get_layer('conv_4').get_weights()[0]

weights_dict = {}
num_filters = len(weights_conv[0, 0, 0, :])
for j in range(num_filters):
    w_s = np.sum(abs(weights_conv[:, :, :, j]))
    filt = f'filt_{j}'
    weights_dict[filt] = w_s

weights_dict_sort = sorted(weights_dict.items(), key=lambda kv: kv[1])
print(weights_dict_sort)

num_channels = 6
layer_4 = model.get_layer('conv_4')
channels_4 = [int(weights_dict_sort[i][0].split('_')[1]) for i in range(num_channels)]
model_new = delete_channels(model, layer_4, channels_4)
model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_new.evaluate(X_test, to_categorical(y_test)))