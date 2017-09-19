"""
From: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
DonkeyCar: https://github.com/wroscoe/donkey/blob/master/donkeycar/parts/ml/keras.py
Keras examples: https://github.com/fchollet/keras/tree/master/examples
"""

import os
import numpy as np
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed

def make_model( num_actions, input_dim, l2_reg=0.005, model_name="orig" ):
    if model_name == "lstm":
        return make_model_lstm( num_actions, input_dim, l2_reg )
    elif model_name == "flat":
        return make_model_flat( num_actions, input_dim, l2_reg )
    elif model_name == "orig":
        return make_model_orig( num_actions, input_dim, l2_reg )
    else:
        print( "Invalid model name. Options are: lstm, flat, orig" )
        return None

def make_model_lstm( num_actions, input_dim, batch_size, timesteps, l2_reg=0.005 ):
    model = Sequential()
    model.add( Flatten(input_shape=input_dim) )
    model.add( LSTM(50, input_shape=(1,1,np.prod(input_dim)), stateful=False, return_sequences=True) )
    #model.add( LSTM(50, input_shape=(1,np.prod(input_dim))) )
    #model.add( TimeDistributed( LSTM(50), input_shape=(batch_size,timesteps,input_dim[0],input_dim[1]) ) )
    model.add( Dense(num_actions, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)) )

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam' )
#                  metrics=['accuracy'])

    return model

def make_model_flat( num_actions, input_dim, l2_reg=0.005 ):
    model = Sequential()
    model.add(Flatten(input_shape=input_dim))
    model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    #model.add(Dropout(0.5))
    model.add(Dense(num_actions, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam' )
#                  metrics=['accuracy'])

    return model

def make_model_orig( num_actions, input_dim, l2_reg=0.005 ):
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
                  optimizer='adam' )
#                  metrics=['accuracy'])

    return model

def build_policy_and_value_networks(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
        
        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        shared = Convolution2D(name="conv1", nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        shared = Convolution2D(name="conv2", nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(shared)
        shared = Flatten()(shared)
        shared = Dense(name="h1", output_dim=256, activation='relu')(shared)

        action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(shared)
        
        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        policy_network = Model(input=inputs, output=action_probs)
        value_network = Model(input=inputs, output=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(state)
        v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params

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

if __name__ == "__main__":
#    model = make_model( 6, (84,84,4), model_name="flat")
#    model.summary()
#    print("")
#    model = make_model( 6, (84,84,4), model_name="orig")
#    model.summary()
    model = make_model_lstm( 6, (84,84), 32, 10 )
    model.summary()
