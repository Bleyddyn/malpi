"""
From: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
DonkeyCar: https://github.com/wroscoe/donkey/blob/master/donkeycar/parts/ml/keras.py
Keras examples: https://github.com/fchollet/keras/tree/master/examples
"""

import os
import numpy as np
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import regularizers
from keras.models import model_from_json

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

def save_model( model, model_dir, model_name ):
    filename = os.path.join( model_dir, model_name + ".hdf5" )
    model.save_weights(filename)
    filename = os.path.join( model_dir, model_name + ".json" )
    with open(filename, "w") as outfile:
        json.dump(model.to_json(), outfile)

def read_model( model_dir, model_name ):
    filename = os.path.join( model_dir, model_name+'.json')
    with open(filename, "r") as jfile:
        model = model_from_json(json.load(jfile))

    filename = os.path.join( model_dir, model_name+'.hdf5')
    model.load_weights( filename )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
