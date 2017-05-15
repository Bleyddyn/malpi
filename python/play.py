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
import time
from collections import deque, namedtuple
import copy
from scipy.misc import imresize

from malpi.layers import *
from malpi.model import *

np.seterr(all='raise')

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
  I = I[35:195] # crop
#  I = I[::2,::2,:] # downsample by factor of 2
  I = imresize(I, (84,84), interp='nearest' )
  I = np.sum( I * rgb_weights, axis=2) # Convert to grayscale, shape = (84,84)
  return I.astype(np.float)

def play(behavior, env, options):
    running_reward = None
    reward_sum = 0
    episode_number = 0
    steps = 0
    act_random = False
    # Atari Breakout Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
    VALID_ACTIONS = xrange(env.action_space.n) # [0, 1, 2, 3]

    observation = env.reset()
    state = prepro(observation)
    state = np.stack([state] * 4, axis=0)

    while True:
      if options.render: env.render()

      if act_random:
          action = np.random.randint(env.action_space.n)
          observation, reward, done, info = env.step(action)
      else:
          q_values, _ = behavior.forward(state.reshape(1,4,84,84), mode="test")
          #print "Action: %s" % ( env.get_action_meanings()[ np.argmax(q_values) ], )
          action = np.argmax(q_values)
          action += 2

          if ksteps > 1:
              reward = 0
              done = False
              next_state = copy.deepcopy(state) # 5.87693955e-05 seconds
              for i in range(ksteps):
                  observation, r, d, info = env.step(action+2)
                  reward += r
                  if d: done = True
                  observation = prepro(observation) #  1.22250773e-03 seconds
                  next_state[i,:,:] = observation
          else:
              observation, reward, done, info = env.step(action+2)
              observation = prepro(observation) #  1.22250773e-03 seconds
              next_state = copy.deepcopy(state) # 5.87693955e-05 seconds
              next_state[0,:,:] = next_state[1,:,:]
              next_state[1,:,:] = next_state[2,:,:]
              next_state[2,:,:] = next_state[3,:,:]
              next_state[3,:,:] = observation # 2.22372381e-05 seconds for all four

      reward_sum += reward
      steps += 1

      if done: # an episode finished
        episode_number += 1

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

        reward_sum = 0
        observation = env.reset()
        if not act_random:
            state = prepro(observation)
            state = np.stack([state] * 4, axis=0)

      if not act_random and reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d, steps %d, reward: %f' % (episode_number, steps, reward))
        steps = 0


def getOptions():
    usage = "Usage: python play.py [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-g","--game", default="Breakout-v0", help="The game environment to use. Defaults to Breakout");
    parser.add_option("-k","--k_steps", type="int", default=4, help="How many game steps to take before the model chooses a new action.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        exit()

    if options.k_steps != 1 and options.k_steps != 4:
        print "Game step sizes other than 1 and 4 are not currently supported."
        exit()

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]

    options.model_name = args[0]
    options.dir_model = os.path.expanduser(options.dir_model)

    return (options, args)

if __name__ == "__main__":
    options, _ = getOptions()

    env = gym.envs.make(options.game)
    print "Actions meanings: %s" % (env.get_action_meanings(),)

    print "Reading model..."
    with open( os.path.join( options.dir_model, options.model_name+'.pickle'), 'rb') as f:
        model = pickle.load( f )
    if not hasattr(model, 'env'):
        print "Warning, model may not work with the current environment."
      
    if hasattr(model, 'env'):
        if model.env != options.game:
            print "Model was not initialized for the current environment: %s vs %s" % (model.env,options.game)
            exit()

    #model.params['b5'][2] += 1
    #model.params['b5'][3] += 1

    play(model, env, options)
