""" Trains an agent with Deep Q Learning or Double DQN on Breakout. Uses OpenAI Gym.
"""
import sys
import os

sys.path.insert(0,os.path.expanduser('~/Library/Python/2.7/lib/python/site-packages/'))

import numpy as np
import cPickle as pickle
import gym
from optparse import OptionParser
import itertools
import random
from collections import deque, namedtuple
import copy
from scipy.misc import imresize
from matplotlib import pyplot as plt
from malpi.model import *

class Experience(object):
    
    def __init__( self, maxN, state_dim ):
        self.N = maxN
        sdim = (maxN,) + state_dim
        self.states = np.zeros(sdim)
        self.actions = np.zeros(maxN)
        self.rewards = np.zeros(maxN)
        self.next_states = np.zeros(sdim)
        self.next_insert = 0
        self.max_batch = 0

    def size( self ):
        return self.max_batch

    def save( self, state, action, reward, next_state ):
        self.states[self.next_insert,:] = state
        self.actions[self.next_insert] = action
        self.rewards[self.next_insert] = reward
        self.next_states[self.next_insert,:] = next_state
        self.next_insert += 1
        self.max_batch = max(self.next_insert, self.max_batch)
        if self.next_insert >= self.N:
            self.next_insert = 0

    def batch( self, batch_size ):
        if batch_size >= self.max_batch:
            return (None,None,None,None)
        start = np.random.randint( 0, high=(1+self.max_batch - batch_size) )
        end = start + batch_size
        r_s = self.states[start:end,:]
        r_a = self.actions[start:end]
        r_r = self.rewards[start:end]
        r_n = self.next_states[start:end,:]
        return (r_s,r_a,r_r,r_n)

    def clear( self ):
        self.next_insert = 0
        self.max_batch = 0

    @staticmethod
    def test():
        e = Experience( 10, (20,20) )
        s = np.zeros( (20,20) )
        e.save( s, 1, -3, s )
        e.save( s, 2, -4, s )
        e.save( s, 3, -5, s )
        e.save( s, 4, -6, s )
        print e.max_batch # 4
        s1, a, r, n = e.batch( 2 )
        print s1.shape # (2, 20, 20)
        print a # e.g. [ 1.  2.]
        print r # e.g. [-3. -4.]
        for _ in range(2):
            e.save( s, 5, -7, s )
            e.save( s, 6, -8, s )
            e.save( s, 7, -9, s )
            e.save( s, 8, -10, s )
        print e.max_batch # 10
        print e.actions[0:2]
        print e.rewards[0:2]

def displayExp():        
    with open( 'experience.pickle', 'rb') as f:
        exp = pickle.load( f )

    for s in range(10,25):
        state = exp.states[s]
        diff1 = np.sum(state[3] - state[0])
        print diff1
        for i in range(state.shape[0]):
            ax1=plt.subplot(4,1,i+1)
            plt.imshow(state[i], cmap='gray', interpolation='nearest')
        plt.show()

def prepro(I):
  """ prepro 210x160x3 uint8 frame into (84x84) float 
      Code from the Denny Britz DQN chapter:
      self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize_images( self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
      self.output = tf.squeeze(self.output)
  """

  rgb_weights = [0.2989, 0.5870, 0.1140]
  I = I[34:194] # crop
#  I = I[::2,::2,:] # downsample by factor of 2
  I = imresize(I, (84,84), interp='nearest' )
  I = np.sum( I * rgb_weights, axis=2) # Convert to grayscale, shape = (84,84)
  return I.astype(np.float)

def initializeModel( name, number_actions ):
    output = "FC-%d" % (number_actions,)
# From the paper, mostly
    layers = ["conv-32", "conv-64", "conv-64", "FC-512", output]
    layer_params = [{'filter_size':8, 'stride':4, 'pad':4 },
        {'filter_size':4, 'stride':2, 'pad':2},
        {'filter_size':3, 'stride':1, 'pad':1},
        {}, {'relu':False} ]
    model = MalpiModel(layers, layer_params, input_dim=(4,84,84), reg=.005, dtype=np.float32, verbose=False)
    model.name = name

    return model

def epsilonGreedy( probabilities, epsilon ):
    if np.random.uniform() < epsilon:
        return np.random.randint( len(probabilities) )
    else:
        return np.argmax(probabilities)

env = gym.make("Pong-v0")
print "Actions: %s" % (env.get_action_meanings(),)

#with open( 'dqn_pong_v1.pickle', 'rb') as f:
#    model = pickle.load( f )

for modnum in range(5):
    model = initializeModel( "model%d" % (modnum,), env.action_space.n )
    action_counts = np.zeros(env.action_space.n)

    for ep in range(5):
        observation = env.reset()
        state = prepro(observation)
        state = np.stack([state] * 4, axis=0)
        done = False
        steps = 0
        while not done:
            aprobs, _ = model.forward( state.reshape(1,4,84,84), mode='test' )
            #print "%d %d" % (np.argmax(aprobs),np.argmax(aprobs[0]))
            aprobs = model.softmax(aprobs[0])
            action = epsilonGreedy(aprobs, 0.5)
            action_counts += aprobs
            #action_counts[action] += 1
            observation, reward, done, info = env.step( action )
            observation = prepro(observation)

            state[0,:,:] = state[1,:,:]
            state[1,:,:] = state[2,:,:]
            state[2,:,:] = state[3,:,:]
            state[3,:,:] = observation
            steps += 1
            #if reward != 0:
            #    print ('ep %d, steps %d, reward: %f' % (ep, steps, reward))

    print action_counts

#for step in range(30):
#    observation, reward, done, info = env.step( np.random.randint(env.action_space.n) )
#    observation = prepro(observation)
#
#    state = np.concatenate( (state[0:3,:,:], observation.reshape(1,84,84)), axis=0)
#    #state[0,:,:] = state[1,:,:]
#    #state[1,:,:] = state[2,:,:]
#    #state[2,:,:] = state[3,:,:]
#    #state[3,:,:] = observation
#
#    diff = np.sum( state[3,:,:] - state[0,:,:] )
#    if( diff > 0.0 ):
#        for i in range(state.shape[0]):
#            ax1=plt.subplot(5,1,i+1)
#            plt.imshow(state[i], cmap='gray', interpolation='nearest')
#        ax1=plt.subplot(5,1,5)
#        plt.imshow(observation, cmap='gray', interpolation='nearest')
#        plt.show()
