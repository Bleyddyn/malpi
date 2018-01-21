import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
from keras.utils import plot_model

def make_model(num_actions, timesteps, input_dim, l2_reg=0.005 ):
    input_shape=(1,None) + input_dim
    model = Sequential()
    model.add(TimeDistributed( Convolution2D(8, (3, 3), strides=(2,2), activation='relu' ), batch_input_shape=input_shape) )
    model.add(TimeDistributed( Convolution2D(16, (3, 3), strides=(2,2), activation='relu', ) ))
    model.add(TimeDistributed( Convolution2D(32, (3, 3), strides=(2,2), activation='relu', ) ))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, return_sequences=True, activation='relu', stateful=True))
    model.add(Dense(num_actions, activation='softmax', ))
    model.compile(loss='categorical_crossentropy', optimizer='adam' )
    return model

batch_size = 16
timesteps = 10
num_actions = 6
model = make_model( num_actions, 1, (84,84,3) )
model.summary()
plot_model(model, to_file='keras_q.png', show_shapes=True)

# Fake training batch. Would be pulled from a replay memory
batch = np.random.uniform( low=0, high=255, size=(batch_size,timesteps,84,84,3) )
y = np.random.randint( 0, high=5, size=(160) )
y = to_categorical( y, num_classes=num_actions )
y = y.reshape( batch_size, timesteps, num_actions )

# Need to find a way to prevent the optimizer from updating every b, but accumulate updates over an entire batch (batch_size).
for b in range(batch_size):
    pred = model.train_on_batch( np.reshape(batch[b,:], (1,timesteps,84,84,3)), np.reshape(y[b,:], (1,timesteps,num_actions)) )
    #for t in range(timesteps):
    #    pred = model.train_on_batch( np.reshape(batch[b,t,:], (1,1,84,84,3)), np.reshape(y[b,t,:], (1,1,num_actions)) )
    model.reset_states() # Don't carry internal state between batches

# move trained network to robot

# This works, but it isn't practical to not get outputs (actions) until after 10 timesteps
#batch = np.random.uniform( low=0, high=255, size=(1,timesteps,84,84,3) )
#pred = model.predict( batch, batch_size=1 )

# This is what I would need to do on my robot, with the LSTM keeping state between calls to predict
max_time = 10 # or 100000, or forever, etc.
for i in range(max_time) :
    image = np.random.uniform( low=0, high=255, size=(1,1,84,84,3) ) # pull one image from camera
    # stateful should be true here
    pred = model.predict( image, batch_size=1 )
    # take action based on pred
    print( pred )
