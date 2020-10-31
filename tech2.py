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

import tensorflow_model_optimization as tfmot
import tempfile

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)
model_for_pruning.summary()

# log_dir = tempfile.mkdtemp()
callbacks = [
             tfmot.sparsity.keras.UpdatePruningStep(),
            #  tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_for_pruning.fit(X_train, to_categorical(y_train, 10), batch_size=32, callbacks=callbacks, epochs=5)

print(model_for_pruning.evaluate(X_test, to_categorical(y_test, 10)))