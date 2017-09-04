"""
From: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
DonkeyCar: https://github.com/wroscoe/donkey/blob/master/donkeycar/parts/ml/keras.py
Keras examples: https://github.com/fchollet/keras/tree/master/examples
"""

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers

from keras.datasets import mnist
 
#from matplotlib import pyplot as plt

def load_mnist():
# Load pre-shuffled MNIST data into train and test sets
    path = os.path.expanduser( "~/.keras/datasets/mnist.npz" )
    f = np.load(path)

    X_train = f[0][0]
    y_train = f[0][1]
    X_test = f[1][0]
    y_test = f[1][1]

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

# Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

#print( X_train.shape )
#plt.imshow(X_train[0])
#plt.show()
#print Y_train.shape

    return X_train, X_test, Y_train, Y_test

def make_model( num_actions, input_dim, l2_reg=0.005 ):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', strides=(1,1), kernel_regularizer=regularizers.l2(l2_reg), input_shape=input_dim))
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', strides=(1,1), kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(num_actions, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = make_model()
    model.summary()
    model.load_weights('tutorial1_ep5.hdf5')

# Save this in an experiment directory so I know exactly what model was used
#config = model.get_config()
# e.g.
#model = Model.from_config(config)
# or, for Sequential:
#model = Sequential.from_config(config)


#print( model.layers )

    X_train, X_test, Y_train, Y_test = load_mnist()
    hist = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.2, validation_data=None, shuffle=True)
# hist = History object with loss/metric data from training

    score = model.evaluate(X_test, Y_test, verbose=0)
    print( "Test score: {}".format(score) )

#scores = model.predict( X_test, batch_size=32, verbose=0)
#scores = model.predict_on_batch( X_test )

    model.save_weights('tutorial1_ep5.hdf5')
#model.load_weights('tutorial1.hdf5')

