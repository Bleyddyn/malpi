""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
    From: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""
import sys
import os

sys.path.insert(0,os.path.expanduser('~/Library/Python/2.7/lib/python/site-packages/'))

import numpy as np
import cPickle as pickle
import gym
from optparse import OptionParser

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
        
def saveModel( model, options ):
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( model, f, pickle.HIGHEST_PROTOCOL)

def initializeModel( name, directory='.' ):
    imsize = 80*80
    layers = ["FC-200", "FC-1"]
    layer_params = [{},{'relu':False}]
#    layers = ["conv-8", "maxpool", "conv-16", "maxpool", "conv-32", "lstml", "FC-5"]
#    layer_params = [{'filter_size':3, 'stride':2, 'pad':1 }, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
#        {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
#        {'filter_size':3}, {'hidden':500}, {} ]
    model = MalpiModel(layers, layer_params, input_dim=(imsize), reg=.005, dtype=np.float32, verbose=True)
    model.name = name

    print
    model.describe()
    print

    return model

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
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

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epx, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def train(model, options):
# hyperparameters
    batch_size = 32 # every how many experience entries to backprop at once
    update_rate = 10 # every how many episodes to do a param update?
    learning_rate = 1e-3 # was: 1e-4
    learning_rate_decay = 1.0 # 0.999
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    render = options.render
    D = np.prod(model.input_dim)

    grad_buffer = { k : np.zeros_like(v) for k,v in model.params.iteritems() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.params.iteritems() } # rmsprop memory

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    running_reward = None
    reward_sum = 0
    episode_number = options.starting_ep
    steps = 0

    test_ob = prepro(observation)
    exp_history = Experience( 2000, test_ob.shape )
    short_history = Experience( 50, test_ob.shape )

    while True:
      if options.render: env.render()

      # preprocess the observation, set input to network to be difference image
      cur_x = prepro(observation)
      x = cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x

      # forward the policy network and sample an action from the returned probability
      x = x.reshape(1,D)
      aprob, cache = model.forward(x, mode='train')
      aprob = sigmoid(aprob)
      action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
      #print "action/aprob: %d/%f" % (action,aprob)

      #y = np.zeros([self.action_size])
      #y[action] = 1
      #self.gradients.append(np.array(y).astype('float32') - prob)

      y = 1 if action == 2 else 0 # a "fake label"
      loss = y - aprob # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward
      steps += 1

      exp_history.save( x, loss, reward, x )
      short_history.save( x, loss, reward, x )

      if done: # an episode finished
        episode_number += 1

        for _ in range(10):
            states, losses, rewards, _ = exp_history.batch( batch_size )
            if states is not None:
                 _, cache = model.forward( states, mode='train' )
                 rewards = discount_rewards( rewards, gamma, normalize=False ) # compute the normalized discounted reward backwards through time
                 if np.count_nonzero(rewards) > 0:
                     losses *= rewards
                     _, grad = model.backward(cache, 0, losses.reshape(batch_size,1) ) # def backward(self, layer_caches, data_loss, dx ):
                     for k in model.params: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every update_rate episodes
        if episode_number % update_rate == 0:
          for k,v in model.params.iteritems():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model.params[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 10 == 0:
            learning_rate *= learning_rate_decay
            if learning_rate_decay < 1.0:
                print "Learning rate: %f" % (learning_rate,)
            saveModel( model, options )
            with open( model.name + '.txt', 'a+') as f:
                f.write( "%d,%f\n" % (episode_number,running_reward) )
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

      if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d, steps %d, reward: %f' % (episode_number, steps, reward)) + ('' if reward == -1 else ' !!!!!!!!')
        steps = 0

        bsize = 49
        if short_history.max_batch < bsize:
            bsize = short_history.max_batch
        states, losses, rewards, _ = short_history.batch( bsize )
        if states is not None:
             _, cache = model.forward( states, mode='train' )
             rewards = discount_rewards( rewards, gamma, normalize=False ) # compute the normalized discounted reward backwards through time
             if np.count_nonzero(rewards) > 0:
                 losses *= rewards
                 _, grad = model.backward(cache, 0, losses.reshape(bsize,1) ) # def backward(self, layer_caches, data_loss, dx ):
                 for k in model.params: grad_buffer[k] += grad[k] # accumulate grad over batch
        short_history.clear()


def getOptions():
    usage = "Usage: python pg-pong [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize model, save to <model name>.pickle, then start training.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-s","--starting_ep", type="int", default=0, help="Starting episode number (for record keeping).");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model, then exit.");

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
#    for k,v in model.params.iteritems():
#        print "%s: %d" % (k, len(v))
    pass

if __name__ == "__main__":
    options, _ = getOptions()

    if options.initialize:
        print "Initializing model..."
        model = initializeModel( options.model_name, options.dir_model )
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

    train(model, options)
