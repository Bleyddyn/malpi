import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils

def make_model( num_actions=4, input_dim=(84,84,4), l2_reg=0.005 ):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', strides=(4,4), input_shape=input_dim))
    model.add(Convolution2D(64, (3, 3), activation='relu', strides=(2,2)))
    model.add(Convolution2D(32, (3, 3), activation='relu', strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def make_model_1( num_actions=4, input_dim=(84,84,4), l2_reg=0.005 ):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), activation='relu', strides=(4,4), input_shape=input_dim))
    model.add(Convolution2D(64, (4, 4), activation='relu', strides=(2,2)))
    model.add(Convolution2D(32, (3, 3), activation='relu', strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = make_model_1()
model.summary()

model = make_model()
model.summary()
