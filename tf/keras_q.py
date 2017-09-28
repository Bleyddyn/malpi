import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical

def make_model(num_actions, timesteps, input_dim, l2_reg=0.005 ):
    input_shape=(timesteps,) + input_dim
    model = Sequential()
    model.add(TimeDistributed( Convolution2D(8, (3, 3), strides=(2,2), activation='relu' ), input_shape=input_shape) )
    model.add(TimeDistributed( Convolution2D(16, (3, 3), strides=(2,2), activation='relu', ) ))
    model.add(TimeDistributed( Convolution2D(32, (3, 3), strides=(2,2), activation='relu', ) ))
    #model.add(TimeDistributed(Flatten()))
    model.add( Reshape( (timesteps,(9*9*32)) ) )
    model.add(LSTM(512, return_sequences=True, activation='relu', unroll=True))
    model.add(Dense(num_actions, activation='softmax', ))
    model.compile(loss='categorical_crossentropy', optimizer='adam' )
    return model

batch_size = 16
timesteps = 10
num_actions = 6
model = make_model( num_actions, timesteps, (84,84,3) )
model.summary()

# Fake training batch. Would be pulled from a replay memory
batch = np.random.uniform( low=0, high=255, size=(batch_size,timesteps,84,84,3) )
y = np.random.randint( 0, high=5, size=(160) )
y = to_categorical( y, num_classes=num_actions )
y = y.reshape( batch_size, timesteps, num_actions )
# stateful should be false here
pred = model.train_on_batch( batch, y )

# move trained network to robot

# This works, but it isn't practical to not get outputs (actions) until after 10 timesteps
batch = np.random.uniform( low=0, high=255, size=(1,timesteps,84,84,3) )
pred = model.predict( batch, batch_size=1 )

# This is what I would need to do on my robot, with the LSTM keeping state between calls to predict
max_time = 10 # or 100000, or forever, etc.
for i in range(max_time) :
    image = np.random.uniform( low=0, high=255, size=(1,1,84,84,3) ) # pull one image from camera
    # stateful should be true here
    pred = model.predict( image, batch_size=1 )
    # take action based on pred
