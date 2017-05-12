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
from malpi.optimizer import Optimizer

#np.seterr(all='raise')
np.seterr(under='ignore')

class Experience(object):
    
    def __init__( self, maxN, state_dim ):
        self.N = maxN
        sdim = (maxN,) + state_dim
        self.states = np.zeros(sdim)
        self.actions = np.zeros(maxN)
        self.rewards = np.zeros(maxN)
        self.done = np.ones(maxN).astype(np.float)
        self.next_states = np.zeros(sdim)
        self.next_insert = 0
        self.max_batch = 0

    def size( self ):
        return self.max_batch

    def save( self, state, action, reward, done, next_state ):
        self.states[self.next_insert,:] = state
        self.actions[self.next_insert] = action
        self.rewards[self.next_insert] = reward
        if done:
            self.done[self.next_insert] = 0.0
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
        r_d = self.done[start:end]
        r_n = self.next_states[start:end,:]
        return (r_s,r_a,r_r,r_d,r_n)

    def clear( self ):
        self.next_insert = 0
        self.max_batch = 0

    @staticmethod
    def test():
        e = Experience( 10, (20,20) )
        s = np.zeros( (20,20) )
        e.save( s, 1, -3, False, s )
        e.save( s, 2, -4, False, s )
        e.save( s, 3, -5, False, s )
        e.save( s, 4, -6, False, s )
        print e.max_batch # 4
        s1, a, r, d, n = e.batch( 2 )
        print s1.shape # (2, 20, 20)
        print a # e.g. [ 1.  2.]
        print r # e.g. [-3. -4.]
        for _ in range(2):
            e.save( s, 5, -7, False, s )
            e.save( s, 6, -8, False, s )
            e.save( s, 7, -9, True, s )
            e.save( s, 8, -10, False, s )
        print e.max_batch # 10
        print e.actions[0:2]
        print e.rewards[0:2]

class Experience2(object):

    Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def __init__( self, maxN, stateDim=None ):
        # stateDim is only for compatibility with Experience
        self.maxN = maxN
        self.memory = []

    def size( self ):
        return len(self.memory)

    def save( self, state, action, reward, done, next_state ):
        if len(self.memory) == self.maxN:
            self.memory.pop(0)

        if done:
            done = 0.0
        else:
            done = 1.0

        self.memory.append(self.Transition(state, action, reward, done, next_state))

    def batch( self, batch_size ):
        if batch_size >= len(self.memory):
            return (None,None,None,None,None)

        samples = random.sample(self.memory, batch_size)
        states_batch, action_batch, reward_batch, done_batch, next_states_batch = map(np.array, zip(*samples))
        return (states_batch, action_batch, reward_batch, done_batch, next_states_batch)


    def clear( self ):
        self.memory = []

    @staticmethod
    def test():
        e = Experience2( 10 )
        s = np.zeros( (20,20) )
        e.save( s, 1, -3, False, s )
        e.save( s, 2, -4, True, s )
        e.save( s, 3, -5, False, s )
        e.save( s, 4, -6, True, s )
        s1, a, r, d, n = e.batch( 2 )
        print s1.shape # (2, 20, 20)
        print a # e.g. [ 1.  2.]
        print r # e.g. [-3. -4.]
        print d
        for _ in range(2):
            e.save( s, 5, -7, False, s )
            e.save( s, 6, -8, False, s )
            e.save( s, 7, -9, True, s )
            e.save( s, 8, -10, False, s )

def stats(arr, msg=""):
    mi = np.min(arr)
    ma = np.max(arr)
    av = np.mean(arr)
    std = np.std(arr)
    abs_arr = np.abs(arr)
    mi_abs = np.min(abs_arr)
    ma_abs = np.max(abs_arr)
    print "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs)
            
def saveModel( model, options ):
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( model, f, pickle.HIGHEST_PROTOCOL)

def initializeModel( name, number_actions ):
    output = "FC-%d" % (number_actions,)
#    layers = ["conv-32", "maxpool", "conv-64", "maxpool", "conv-64", "FC-512", output]
#    layer_params = [{'filter_size':3, 'stride':1 },
#        {'pool_stride': 2, 'pool_width': 2, 'pool_height': 2},
#        {'filter_size':3, 'stride':1 },
#        {'pool_stride': 2, 'pool_width': 2, 'pool_height': 2},
#        {'filter_size':3, 'stride':2 },
#        {}, {'relu':False} ]
# From the DQN paper, mostly
    layers = ["conv-32", "conv-64", "conv-64", "FC-512", output]
    layer_params = [{'filter_size':8, 'stride':4, 'pad':4 },
        {'filter_size':4, 'stride':2, 'pad':2},
        {'filter_size':3, 'stride':1, 'pad':1},
        {}, {'relu':False} ]
    model = MalpiModel(layers, layer_params, input_dim=(4,84,84), reg=0.005, dtype=np.float32, verbose=True)
    model.name = name

    print
    model.describe()
    print

    return model

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into (84x84) float 
  """

  rgb_weights = [0.2989, 0.5870, 0.1140]
  I = I[35:195] # crop
  I = imresize(I, (84,84), interp='nearest' )
  I = np.sum( I * rgb_weights, axis=2) # Convert to grayscale, shape = (84,84)
  return I.astype(np.float) / 255.0
  #return I.astype(np.float)

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
        q_values,_ = estimator.forward(observation, mode="test")
        best_action = np.argmax(q_values[0])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def choose_epsilon_greedy( estimator, observation, epsilon, nA ):
    if np.random.random() < epsilon:
        return np.random.randint(nA)
    else:
        q_values,_ = estimator.forward(observation, mode="test")
        return np.argmax(q_values[0])

def check_weights( model ):
    for k,w in model.params.iteritems():
        smallest = np.min( np.abs(w) )
        print "Smallest %s: %g" % (k,smallest)
        mask_zeros = w != 0.0
        mask = np.abs(w) < 1e-20
        mask = np.logical_and(mask_zeros,mask)
        if np.count_nonzero(mask) > 0:
            print "Underflow in %s " % (k,)

def train(target, env, options):
    batch_size = 32 # backprop batch size
    update_rate = 2 # every how many episodes to copy behavior model to target
    learning_rate = 0.005
    learning_rate_decay = 1.0 # 0.999
    gamma = 0.99 # discount factor for reward
    epsilon = 0.2
    ksteps = options.k_steps # number of frames to skip before selecting a new action

    target.reg = 0.005

    behavior = copy.deepcopy(target)
    optim = Optimizer( "rmsprop", behavior, learning_rate=0.005) # learning_rate = 0.001, decay_rate=0.9, epsilon=1e-8 )
    #policy = make_epsilon_greedy_policy(behavior, epsilon, env.action_space.n)

    running_reward = None
    reward_sum = 0
    episode_number = options.starting_ep
    steps = 0
    episode_steps = 0
    point = 1 # how many points have been scored by either side in this episode

    num_actions = env.action_space.n
    VALID_ACTIONS = xrange(num_actions) # [0, 1, 2, 3]

    observation = env.reset()
    state = prepro(observation)
    state = np.stack([state] * 4, axis=0)

    exp_history = Experience2( 2000, state.shape )

    with open( target.name + '_hparams.txt', 'a+') as f:
        f.write( "%s = %s\n" % ('Start',time.strftime("%Y-%m-%d %H:%M:%S")) )
        f.write( "%s = %d\n" % ('batch_size',batch_size) )
        f.write( "%s = %d\n" % ('update_rate',update_rate) )
        f.write( "%s = %f\n" % ('gamma',gamma) )
        f.write( "%s = %f\n" % ('epsilon',epsilon) )
        f.write( "%s = %d\n" % ('k-steps',ksteps) )
        f.write( "Optimizer %s\n" % (optim.optim_type,) )
        f.write( "   %s = %f\n" % ('learning rate',optim.learning_rate) )
        f.write( "   %s = %f\n" % ('decay rate',optim.decay_rate) )
        f.write( "   %s = %f\n" % ('epsilon',optim.epsilon) )
        f.write( "\n" )

    while True:
      if options.render: env.render()

      action = choose_epsilon_greedy( behavior, state.reshape(1,4,84,84), epsilon, num_actions )
      if epsilon > 0.1:
          epsilon -= 9e-07 # Decay to 0.1 over 1 million steps

      # step the environment once, or ksteps times
      # TODO: fix this so it will work for ksteps other than 1 and 4
      if ksteps > 1:
          reward = 0
          done = False
          next_state = copy.deepcopy(state) # 5.87693955e-05 seconds
          for i in range(ksteps):
              observation, r, d, info = env.step(action)
              reward += r
              if d: done = True
              observation = prepro(observation) #  1.22250773e-03 seconds
              next_state[i,:,:] = observation
      else:
          observation, reward, done, info = env.step(action)
          observation = prepro(observation) #  1.22250773e-03 seconds
          next_state = copy.deepcopy(state) # 5.87693955e-05 seconds
          next_state[0,:,:] = next_state[1,:,:]
          next_state[1,:,:] = next_state[2,:,:]
          next_state[2,:,:] = next_state[3,:,:]
          next_state[3,:,:] = observation # 2.22372381e-05 seconds for all four

      reward_sum += reward
      steps += ksteps
      episode_steps += ksteps

      exp_history.save( state, action, reward, done, next_state ) # 2.91559257e-04 seconds
      state = next_state

      if exp_history.size() > (batch_size * 5):
          states, actions, rewards, batch_done, new_states = exp_history.batch( batch_size ) # 3.04588500e-05 seconds
          actions = actions.astype(np.int)

          #onehot = np.zeros( (batch_size,actions.shape[0]) )
          #onehot[np.arange(batch_size),actions] = 1.0
          #print np.mean(onehot, axis=0)[0:6]

          #stats(states,"states ")
          #stats(actions,"actions ")
          #stats(rewards,"rewards ")
          #stats(batch_done,"batch_done ")
          #stats(new_states,"new_states ")

          # Save one mini-batch for testing
          #with open('one_experience.pickle', 'wb') as pf:
          #    one = (states,actions,rewards,batch_done,new_states)
          #    pickle.dump( one, pf, pickle.HIGHEST_PROTOCOL)

          target_values, _ = target.forward( new_states, mode='test' ) # 2.00298658e-01 seconds

          double_dqn = True
          if double_dqn:
              behavior_values, _ = behavior.forward( new_states, mode='test' ) # 1.74144219e-01 seconds
              best_actions = np.argmax(behavior_values,axis=1)
              q_target = rewards + batch_done * gamma * target_values[np.arange(batch_size), best_actions]
          else:
              q_target = rewards + batch_done * gamma * np.max(target_values, axis=1)

          action_values, cache = behavior.forward(states, mode='train', verbose=False)

          q_error = np.zeros( action_values.shape )
          #q_error[ np.arange(batch_size), 2 ] = -batch_size
          # Only update values for actions taken
          q_error[ np.arange(batch_size), actions ] = q_target - action_values[ np.arange(batch_size), actions ]
          #q_error[ np.arange(batch_size), actions ] = action_values[ np.arange(batch_size), actions ] - q_target
          #q_error /= batch_size
          # See: https://zhuanlan.zhihu.com/p/25771039
          dx = q_error
          dx /= batch_size

          print_stats = False
          if done:
              print_stats = True
              stats( dx, "dx " )

          #print "actions: %s" % (actions[1:5],)
          #print "a_values: %s" % (action_values[1:5,:],)
          #print "t_values: %s" % (target_values[1:5,:],)
          #print "rewards : %s" % (rewards[1:5],)
          #print "batch_d : %s" % (batch_done[1:5],)
          #print "q_target: %s" % (q_target[1:5],)
          #print "q_error : %s" % (q_error[1:5,:],)

          q_error = np.sum( np.square( q_error ) )

          # dx needs to have shape(batch_size,num_actions), e.g. (32,6)
          #check_weights( behavior )
          _, grad = behavior.backward(cache, q_error, dx ) # 2.37275421e-01 seconds
          optim.update( grad, check_ratio=print_stats ) # 1.85747565e-01 seconds

          # Clip very small weights to prevent underflow in multiplications
          if False:
              for k,v in behavior.params.iteritems():
                  mask_zeros = behavior.params[k] != 0.0
                  mask = np.abs(behavior.params[k]) < 1e-15
                  mask = np.logical_and(mask_zeros,mask)
                  behavior.params[k][mask] = 0.0
                  if np.count_nonzero(mask) > 0:
                      print "Underflow in %s " % (k,)

          #stats(states,"states " )
          #stats(reward, "reward " )
          #stats(values, "target " )
          #target.describe()
          #stats(temp_values, "behavior " )
          #stats(q_target, "q_target " )
          #stats(actions, "actions " )
          #print q_error.shape
          #stats(dx, "dx " )
          #print "========"

      if done: # an episode finished
        episode_number += 1
        point = 0

        #At update rate, copy behavior into target
        if episode_number % update_rate == 0:
            target = copy.deepcopy(behavior)
            print "Copying behavior network to target"

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f.  In %d steps' % (reward_sum, running_reward, episode_steps)
        behavior.describe()
        optim.describe()

        if episode_number % 10 == 0:
            optim.learning_rate *= learning_rate_decay
            if learning_rate_decay < 1.0:
                print "Learning rate: %f" % (optim.learning_rate,)
            saveModel( target, options )
            with open( target.name + '.txt', 'a+') as f:
                f.write( "%d,%f\n" % (episode_number,running_reward) )

        reward_sum = 0
        episode_steps = 0
        observation = env.reset()
        state = prepro(observation)
        state = np.stack([state] * 4, axis=0)

      if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        point += 1
        print ('ep %d, points %d, steps %d, reward: %f' % (episode_number, point, steps, reward))
        steps = 0


def getOptions():
    usage = "Usage: python pg-pong [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize model, save to <model name>.pickle, then start training.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-s","--starting_ep", type="int", default=0, help="Starting episode number (for record keeping).");
    parser.add_option("-k","--k_steps", type="int", default=4, help="How many game steps to take before the model chooses a new action.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model, then exit.");
    parser.add_option("-g","--game", default="Breakout-v0", help="The game environment to use. Defaults to Breakout");

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

def test(options, model):
    Experience2.test()

#    behavior = copy.deepcopy(model)
#    print "%f == %f" % (model.params['W1'][0,0], behavior.params['W1'][0,0])
#    model.params['W1'][0,0] = 99
#    print "%f == %f" % (model.params['W1'][0,0], behavior.params['W1'][0,0])
#    model.describe()
#    behavior.describe()

#    for k,v in model.params.iteritems():
#        print "%s: %d" % (k, len(v))

#    optim = Optimizer( "rmsprop", model, learning_rate=0.0003, decay_rate=0.99, epsilon=1e-7) # learning_rate = 0.001, decay_rate=0.9, epsilon=1e-8 )
#    optim.describe()

    pass

if __name__ == "__main__":
    options, _ = getOptions()

    env = gym.envs.make(options.game)
    print env.get_action_meanings()

    if options.initialize:
        print "Initializing model with %d actions..." % (env.action_space.n,)
        model = initializeModel( options.model_name, env.action_space.n )
        model.env = options.game
        saveModel( model, options )
    else:
        print "Reading model..."
        with open( os.path.join( options.dir_model, options.model_name+'.pickle'), 'rb') as f:
            model = pickle.load( f )
        if not hasattr(model, 'env'):
            print "Warning, model may not work with the current environment."
      
    if options.desc:
        model.describe()
        exit()

    if options.test_only:
        test(options, model)
        exit()

    if hasattr(model, 'env'):
        if model.env != options.game:
            print "Model was not initialized for the current environment: %s vs %s" % (model.env,options.game)
            exit()

    train(model, env, options)
