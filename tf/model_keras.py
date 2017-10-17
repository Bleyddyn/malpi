"""
From: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
DonkeyCar: https://github.com/wroscoe/donkey/blob/master/donkeycar/parts/ml/keras.py
Keras examples: https://github.com/fchollet/keras/tree/master/examples

Possible method using for loops: https://stackoverflow.com/questions/42629530/obtain-output-at-each-timestep-in-keras-lstm
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

This looks good, except that he states it doesn't work :(
https://github.com/dat-ai/steering-model/blob/master/model/DatNet.py
"""

import os
import numpy as np
import json
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics

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

def make_model_lstm( num_actions, input_dim, batch_size=1, timesteps=None, stateful=True, l2_reg=0.005, optimizer=None, dropouts=[0.25,0.25,0.25,0.25,0.25] ):
    input_shape=(batch_size,timesteps) + input_dim
    model = Sequential()
    model.add(TimeDistributed( Dropout(dropouts[0]), batch_input_shape=input_shape, name="Dropout1") )
    model.add(TimeDistributed( Convolution2D(16, (8, 8), padding='same', strides=(4,4), activation='relu', kernel_regularizer=regularizers.l2(l2_reg) ), name="Conv-8-16" ) )
    model.add(TimeDistributed( Dropout(dropouts[1]), name="Dropout2" ))
    model.add(TimeDistributed( Convolution2D(32, (4, 4), padding='same', strides=(2,2), activation='relu',  kernel_regularizer=regularizers.l2(l2_reg) ), name="Conv-4-32" ))
    model.add(TimeDistributed( Dropout(dropouts[2]), name="Dropout3" ))
    model.add(TimeDistributed( Convolution2D(64, (3, 3), padding='same', strides=(1,1), activation='relu',  kernel_regularizer=regularizers.l2(l2_reg)), name="Conv-3-64" ))
    model.add(TimeDistributed( Dropout(dropouts[3]), name="Dropout4" ))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, activation='relu', stateful=stateful,  kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(TimeDistributed( Dropout(dropouts[4]), name="Dropout5" ))
    model.add(Dense(num_actions, activation='softmax',  kernel_regularizer=regularizers.l2(l2_reg), name="Output" ))
    
    if optimizer is None:
        optimizer = optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy] )

    return model

def make_model_lstm_fit( num_actions, input_dim, batch_size=1, timesteps=None, stateful=False, l2_reg=0.005, optimizer=None, dropouts=[0.25,0.25,0.25,0.25,0.25] ):
    input_shape=(timesteps,) + input_dim
    model = Sequential()
    model.add(TimeDistributed( Dropout(dropouts[0]), input_shape=input_shape, name="Dropout1") )
    model.add(TimeDistributed( Convolution2D(16, (8, 8), padding='same', strides=(4,4), activation='relu', kernel_regularizer=regularizers.l2(l2_reg) ), name="Conv-8-16" ) )
    model.add(TimeDistributed( Dropout(dropouts[1]), name="Dropout2" ))
    model.add(TimeDistributed( Convolution2D(32, (4, 4), padding='same', strides=(2,2), activation='relu',  kernel_regularizer=regularizers.l2(l2_reg) ), name="Conv-4-32" ))
    model.add(TimeDistributed( Dropout(dropouts[2]), name="Dropout3" ))
    model.add(TimeDistributed( Convolution2D(64, (3, 3), padding='same', strides=(1,1), activation='relu',  kernel_regularizer=regularizers.l2(l2_reg)), name="Conv-3-64" ))
    model.add(TimeDistributed( Dropout(dropouts[3]), name="Dropout4" ))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, activation='relu', stateful=stateful,  kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(TimeDistributed( Dropout(dropouts[4]), name="Dropout5" ))
    model.add(Dense(num_actions, activation='softmax',  kernel_regularizer=regularizers.l2(l2_reg), name="Output" ))
    
    if optimizer is None:
        optimizer = optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy] )

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

def make_model_test( num_actions, input_dim, l2_reg=0.005 ):
    model = Sequential()
    model.add(Dropout(0.25, input_shape=input_dim))
    model.add(Convolution2D(16, (8, 8), activation='relu', padding='same', strides=(4,4), kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, (4, 4), activation='relu', padding='same', strides=(2,2), kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(1,1), kernel_regularizer=regularizers.l2(l2_reg)))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(0.25))
    model.add(Dense(num_actions, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))

    optim = optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.005)

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=[metrics.categorical_accuracy] )

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

    model.compile(loss='categorical_crossentropy', optimizer='adam' )

    return model

def make_model_doom(num_actions, timesteps, input_dim, l2_reg=0.005 ):
    # From https://github.com/itaicaspi/keras-dqn-doom/blob/833a56dd13266977c1b1d0083e3827027cd140fb/main.py
    input_shape=(timesteps,) + input_dim
    model = Sequential()
    model.add(TimeDistributed( Convolution2D(8, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_reg) ), input_shape=input_shape) )
    model.add(TimeDistributed( Convolution2D(16, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)) ))
    model.add(TimeDistributed( Convolution2D(32, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)) ))
    #model.add(TimeDistributed(Flatten()))
    model.add( Reshape( (timesteps,(9*9*32)) ) )
    model.add(LSTM(512, return_sequences=True, activation='relu', unroll=True))
    model.add(Dense(num_actions, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))
    model.compile(loss='categorical_crossentropy', optimizer='adam' )
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
    #model = make_model( 6, (84,84,4), model_name="flat")
    #model.summary()
    #model = make_model( 6, (84,84,4), model_name="orig")
    #model.summary()
    #model = make_model_doom( 6, 10, (84,84,3) )
    #model.summary()

    model = make_model_test( 6, (120,120,3), l2_reg=0.005 )
    model.summary()

    model = make_model_lstm( 6, (120,120,3), l2_reg=0.005 )
    model.summary()
    exit()

    model_name = "test_speed"
    t_start = time.time()
    if False and os.path.exists( model_name + ".hdf5" ):
        print( "Reading model" )
        model = read_model( ".", model_name )
    else:
        print( "Making model" )
        #model = make_model_lstm( 6, (84,84), 10 )
        model = make_model_doom( 6, 10, (84,84,3) )
        save_model( model, ".", model_name )
    print( "   {} seconds".format( time.time() - t_start ) )

    model.summary()

    batch = np.random.uniform( low=0, high=255, size=(16,10,84,84,3) )
    y = np.random.randint( 0, high=5, size=(160) )
    y = to_categorical( y, num_classes=6 )
    y = y.reshape( 16, 10, 6 )
# stateful should be false here
    pred = model.train_on_batch( batch, y )

    # This works, but...
    batch = np.random.uniform( low=0, high=255, size=(1,10,84,84,3) )
    pred = model.predict( batch, batch_size=1 )
    print( pred )

    # This is what I would need to do on my robot, with the LSTM keeping state between calls
# stateful should be true here
    batch = np.random.uniform( low=0, high=255, size=(1,1,84,84,3) )
    pred = model.predict( batch, batch_size=1 )
    print( pred )

    t_start = time.time()
    batch = np.random.uniform( low=0, high=255, size=(1,10,84,84,3) )
    for i in range(10):
        pred = model.predict( batch, batch_size=1 )

    print( "10 cycles in {} seconds".format( time.time() - t_start ) )

    # lstm with 200 nodes: 10 cycles in 1.6962685585021973 seconds
# lstm with 2 conv layers (strides 2): 10 cycles in 8.055421829223633 seconds
# lstm with 3 conv layers (strides 2): 10 cycles in 2.724311590194702 seconds


