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
        
class Optimizer(object):
    """ TODO: Add support for Adam: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
    """
    def __init__( self, optim_type, model, learning_rate = 0.001, decay_rate=0.9, epsilon=1e-5 ):
        supported = ["rmsprop"]
        if optim_type not in supported:
            print "Invalid optimizer type: " % (optim_type,)
            print "Supported types: %s" % (str(supported),)
            return

        self.model = model
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = { k : np.zeros_like(v) for k,v in self.model.params.iteritems() }

    def update( self, grad_buffer ):
        for k,v in self.model.params.iteritems():
            g = grad_buffer[k] # gradient
            self.cache[k] = self.decay_rate * self.cache[k] + (1 - self.decay_rate) * g**2
            model.params[k] -= self.learning_rate * g / np.sqrt(self.cache[k] + self.epsilon)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer


def saveModel( model, options ):
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( model, f, pickle.HIGHEST_PROTOCOL)

def initializeModel( name, number_actions ):
    output = "FC-%d" % (number_actions,)
# From the paper, mostly
    layers = ["conv-32", "conv-64", "conv-64", "FC-512", output]
    layer_params = [{'filter_size':8, 'stride':4, 'pad':4 },
        {'filter_size':4, 'stride':2, 'pad':2},
        {'filter_size':3, 'stride':1, 'pad':1},
        {}, {'relu':False} ]
    model = MalpiModel(layers, layer_params, input_dim=(4,80,80), reg=.005, dtype=np.float32, verbose=True)
    model.name = name

    print
    model.describe()
    print

    return model

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
      Code from the Denny Britz DQN chapter:
      self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize_images( self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
      self.output = tf.squeeze(self.output)
  """

  rgb_weights = [0.2989, 0.5870, 0.1140]
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I = np.sum( I * temp.rgb_weights, axis=2) # Convert to grayscale, shape = (80,80)
  return I.astype(np.float).ravel()

def discount_rewards(r, gamma, normalize=True):
    """ take 1D float array of rewards and compute discounted reward.
        if normalize is True: subtract mean and divide by std dev
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    if normalize:
# standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

    return discounted_r

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def train(model, env, options):
    batch_size = 32 # every how many experience entries to backprop at once
    update_rate = 10 # every how many episodes to do a param update?
    learning_rate_decay = 1.0 # 0.999
    gamma = 0.99 # discount factor for reward
    epsilon = 0.1
    render = options.render

    optim = Optimizer( "rmsprop", model, learning_rate = 0.001, decay_rate=0.9, epsilon=1e-8 )
    behavior = copy.deepcopy(model)
    policy = make_epsilon_greedy_policy(behavior, epsilon, env.action_space.n):

    D = np.prod(model.input_dim)

    grad_buffer = { k : np.zeros_like(v) for k,v in model.params.iteritems() } # update buffers that add up gradients over a batch

    observation = env.reset()
    prev_x = None # used in computing the difference frame
    running_reward = None
    reward_sum = 0
    episode_number = options.starting_ep
    steps = 0

    # Atari Breakout Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
    VALID_ACTIONS = xrange(env.action_space.n) # [0, 1, 2, 3]

    test_ob = prepro(observation)
    exp_history = Experience( 2000, test_ob.shape )

    while True:
      if options.render: env.render()

      # preprocess the observation, set input to network to be difference image
      cur_x = prepro(observation)
      x = cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x

      aprob = policy(state)
      action = np.random.choice(VALID_ACTIONS, p=aprob)

      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward
      steps += 1

      exp_history.save( x, loss, reward, observation )
      if exp_history.size() > batch_size:
          states, losses, rewards, new_states = exp_history.batch( batch_size )
          _, cache = model.forward( states, mode='train' )
          rewards = discount_rewards( rewards, gamma, normalize=False ) # compute the normalized discounted reward backwards through time
          q_target = reward + gamma * np.max(behavior.forward(new_states))
          q_error = q_target - model.forward(states)
          _, grad = model.backward(cache, 0, q_error.reshape(batch_size,1) ) # def backward(self, layer_caches, data_loss, dx ):
          for k in model.params: grad_buffer[k] += grad[k] # accumulate grad over batch

      if done: # an episode finished
        episode_number += 1

        # PERFORM RMSPROP PARAMETER UPDATE EVERY UPDAte_rate episodes
        if episode_number % update_rate == 0:
            optim.update( grad_buffer )

        # At some rate, copy model into behavior

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 10 == 0:
            optim.learning_rate *= learning_rate_decay
            if learning_rate_decay < 1.0:
                print "Learning rate: %f" % (optim.learning_rate,)
            saveModel( model, options )
            with open( model.name + '.txt', 'a+') as f:
                f.write( "%d,%f\n" % (episode_number,running_reward) )

        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

      if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d, steps %d, reward: %f' % (episode_number, steps, reward)) + ('' if reward == -1 else ' !!!!!!!!')
        steps = 0


def getOptions():
    usage = "Usage: python pg-pong [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize model, save to <model name>.pickle, then start training.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-s","--starting_ep", type="int", default=0, help="Starting episode number (for record keeping).");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model, then exit.");
    parser.add_option("-g","--game", default="Breakout-v0", help="The game environment to use. Defaults to Breakout");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        exit()

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]

    options.model_name = args[0]
    options.dir_model = os.path.expanduser(options.dir_model)

    return (options, args)

def test(options, model):
    Experience.test()

#    behavior = copy.deepcopy(model)
#    print "%f == %f" % (model.params['W1'][0,0], behavior.params['W1'][0,0])
#    model.params['W1'][0,0] = 99
#    print "%f == %f" % (model.params['W1'][0,0], behavior.params['W1'][0,0])
#    model.describe()
#    behavior.describe()

#    for k,v in model.params.iteritems():
#        print "%s: %d" % (k, len(v))

    pass

if __name__ == "__main__":
    options, _ = getOptions()

    env = gym.envs.make(options.game)

    if options.initialize:
        print "Initializing model..."
        model = initializeModel( options.model_name, env.action_space.n )
        saveModel( model, options )
    else:
        print "Reading model..."
        with open( os.path.join( options.dir_model, options.model_name+'.pickle'), 'rb') as f:
            model = pickle.load( f )
      
    if options.desc:
        model.describe()
        exit()

    if options.test_only:
        test(options, model)
        exit()

    train(model, env, options)
