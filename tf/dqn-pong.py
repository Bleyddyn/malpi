""" Trains an agent with Deep Q Learning or Double DQN on Breakout. Uses OpenAI Gym.
"""
import sys
import os

import numpy as np
import pickle
import gym
from optparse import OptionParser
import itertools
import random
import time
from collections import deque
import copy
from scipy.misc import imresize
import tensorflow as tf

#from malpi.layers import *
#from malpi.model import *
#from malpi.optimizer import Optimizer
from malpi.experience import Experience2

from model_keras import make_model, read_model, save_model
from keras.models import clone_model
 
#np.seterr(all='raise')
np.seterr(under='ignore')

def stats(arr, msg=""):
    mi = np.min(arr)
    ma = np.max(arr)
    av = np.mean(arr)
    std = np.std(arr)
    abs_arr = np.abs(arr)
    mi_abs = np.min(abs_arr)
    ma_abs = np.max(abs_arr)
    print( "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs) )
            
def initializeModel( name, number_actions, input_dim=(84,84,4) ):
    model = make_model( number_actions, input_dim )
    model.name = name
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

def choose_epsilon_greedy( estimator, observation, epsilon, nA ):
    if np.random.random() < epsilon:
        return np.random.randint(nA)
    else:
        q_values = estimator.predict_on_batch( observation )
        return np.argmax(q_values[0])

def check_weights( model ):
    for k,w in model.params.iteritems():
        smallest = np.min( np.abs(w) )
        print( "Smallest %s: %g" % (k,smallest) )
        mask_zeros = w != 0.0
        mask = np.abs(w) < 1e-20
        mask = np.logical_and(mask_zeros,mask)
        if np.count_nonzero(mask) > 0:
            print( "Underflow in %s " % (k,) )

def test(tmodel, env, options):
    reward_100 = 0
    count = 10.0
    epsilon = 0.5
    nA = env.action_space.n

    for i in range(int(count)):
        episode_reward = 0
        state = env.reset()
        state = prepro(state)
        state = np.stack([state] * 4, axis=0)
        done = False
        steps = 0
        while not done:
            if options.render: env.render()
            action = choose_epsilon_greedy( tmodel, state.reshape(1,84,84,4), epsilon, nA )
            if options.k_steps > 1:
                reward = 0
                done = False
                next_state = copy.deepcopy(state)
                for i in range(options.k_steps):
                    observation, r, d, info = env.step(action)
                    reward += r
                    if d: done = True
                    observation = prepro(observation)
                    next_state[i,:,:] = observation
            else:
                observation, reward, done, info = env.step(action)
                observation = prepro(observation)
                next_state = copy.deepcopy(state)
                next_state[0,:,:] = next_state[1,:,:]
                next_state[1,:,:] = next_state[2,:,:]
                next_state[2,:,:] = next_state[3,:,:]
                next_state[3,:,:] = observation # 2.22372381e-05 seconds for all four

            state = next_state
            episode_reward += reward
            steps += 1

        reward_100 += episode_reward
    return (reward_100 / count)

def train(behavior, env, options):
    batch_size = 32 # backprop batch size
    update_rate = 10 # every how many episodes to copy behavior model to target
    gamma = 0.99 # discount factor for reward
    epsilon = 1.0
    epsilon_decay = 0.9999
    ksteps = options.k_steps # number of frames to skip before selecting a new action
    learning_rate = 0.01
    learning_rate_decay = 0.9999
    lr_decay_on_best = 0.95
    clip_error = True

    #target.reg = 0.005

    target = initializeModel( options.model_name, env.action_space.n )
    target_weights = target.trainable_weights
    behavior_weights = behavior.trainable_weights

    # Define target network update operation
    update_target = [target_weights[i].assign(behavior_weights[i]) for i in range(len(target_weights))]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Initialize target network
    sess.run(update_target)

    running_reward = None
    reward_sum = 0
    episode_number = options.starting_ep
    steps = 0
    episode_steps = 0
    point = 1 # how many points have been scored by either side in this episode
    num_actions = env.action_space.n
    action_counts = np.zeros(env.action_space.n)

    reward_100 = deque(maxlen=100)
    if options.play:
        best_test = -21.0
    else:
        #best_test = test(target, env, options)
        best_test = -21.0
        print( "Starting test score: %f" % (best_test,) )

    observation = env.reset()
    state = prepro(observation)
    state = np.stack([state] * 4, axis=0)

    exp_history = Experience2( 2000, state.shape )

    if not options.play:
        with open( os.path.join( options.game + ".txt" ), 'a+') as f:
            f.write( "%s = %s\n" % ('Start',time.strftime("%Y-%m-%d %H:%M:%S")) )
            f.write( "%s = %s\n" % ('Model Name',target.name) )
            f.write( "%s = %d\n" % ('batch_size',batch_size) )
            f.write( "%s = %d\n" % ('update_rate',update_rate) )
            f.write( "%s = %f\n" % ('gamma',gamma) )
            f.write( "%s = %f\n" % ('epsilon',epsilon) )
            f.write( "%s = %f\n" % ('epsilon_decay',epsilon_decay) )
            f.write( "%s = %d\n" % ('k-steps',ksteps) )
            f.write( "%s = %f\n" % ('learning_rate',learning_rate) )
            f.write( "%s = %f\n" % ('learning_rate_decay',learning_rate_decay) )
            f.write( "%s = %f\n" % ('lr_decay_on_best',lr_decay_on_best) )
            f.write( "%s = %s\n" % ('clip_error',str(clip_error)) )
            f.write( "\n" )

    while (options.max_episodes == 0) or (episode_number < options.max_episodes):
        if options.render: env.render()
  
        action = choose_epsilon_greedy( behavior, state.reshape(1,84,84,4), epsilon, num_actions )
        #action = np.random.randint(num_actions)
        action_counts[action] += 1
  
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
 
        if not options.play: 
            exp_history.save( state.reshape(84,84,4), action, reward, done, next_state.reshape(84,84,4) )
        state = next_state
  
        if not options.play and (exp_history.size() > (batch_size * 5)):
            states, actions, rewards, batch_done, new_states, _ = exp_history.batch( batch_size )
            actions = actions.astype(np.int)
  
            target_values = target.predict_on_batch( states )
  
            double_dqn = True
            if double_dqn:
                behavior_values = behavior.predict_on_batch( new_states )
                best_actions = np.argmax(behavior_values,axis=1)
                q_target = rewards + batch_done * gamma * target_values[np.arange(batch_size), best_actions]
            else:
                q_target = rewards + batch_done * gamma * np.max(target_values, axis=1)
  
            action_values = behavior.predict_on_batch(states)
  
            q_error = np.zeros( action_values.shape )
            q_error[ np.arange(batch_size), actions ] = action_values[ np.arange(batch_size), actions ] - q_target
            dx = q_error
            dx /= batch_size
            if clip_error:
                np.clip( dx, -1.0, 1.0, dx )
  
            #print( "actions: %s" % (actions[1:5],) )
            #print( "a_values: %s" % (action_values[1:5,:],) )
            #print( "t_values: %s" % (target_values[1:5,:],) )
            #print( "rewards : %s" % (rewards[1:5],) )
            #print( "batch_d : %s" % (batch_done[1:5],) )
            #print( "q_target: %s" % (q_target[1:5],) )
            #print( "q_error : %s" % (q_error[1:5,:],) )
  
            q_error = np.sum( np.square( q_error ) )

            loss = behavior.train_on_batch(states, dx)
                         
            #stats(states,"states " )
            #stats(reward, "reward " )
            #stats(values, "target " )
            #target.describe()
            #stats(temp_values, "behavior " )
            #stats(q_target, "q_target " )
            #stats(actions, "actions " )
            #print( q_error.shape )
            #stats(dx, "dx " )
            #print( "========" )
  
        if done: # an episode finished
            episode_number += 1
            point = 0
   
            reward_100.append(reward_sum)
  
            print( 'Reward for Ep %d %0.2f  %0.2f' % ( episode_number, reward_sum, np.mean(reward_100) ) )
                                                                                
            if not options.play:
                if episode_number % update_rate == 0:
                    with open( os.path.join( options.dir_model, target.name+ ".txt" ), 'a+') as f:
                        f.write( "%d,%f\n" % (episode_number, np.mean(reward_100) ) )

                    sess.run(update_target)
                    save_model( target, options.dir_model, options.model_name )

                    treward = np.mean(reward_100) # test(target, env, options)

                    print('')
                    print( 'Ep %d' % ( episode_number, ) )
                    print( 'Reward       : %0.2f  %0.2f' % ( reward_sum, np.mean(reward_100) ) )
                    print( "Test reward  : %0.2f vs %0.2f" % (treward, best_test) )
                    print( "Epsilon      : %f" % (epsilon,) )

                    if treward > best_test:
                        best_test = treward
                        save_model( target, options.dir_model, options.model_name + "_best" )
                        print( "Saving current best model." )

                        if treward > 20.5:
                            print( "WON! In %d episodes" % (episode_number,) )
                            break
                if epsilon > 0.1:
                    epsilon *= epsilon_decay

            action_counts = np.zeros(env.action_space.n)
            reward_sum = 0
            episode_steps = 0
            observation = env.reset()
            state = prepro(observation)
            state = np.stack([state] * 4, axis=0)
  
#        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
#          point += 1
#          print( ('ep %d, points %d, steps %d, reward: %f' % (episode_number, point, steps, reward)) )
#          steps = 0

    if not options.play:
        with open( os.path.join( options.game + ".txt" ), 'a+') as f:
            f.write( "%s = %f\n" % ('Final epsilon', epsilon) )
            f.write( "%s = %f\n" % ('Best test score', best_test) )
            f.write( "%s = %d\n" % ('Episodes', episode_number) )
            f.write( "\n\n" )

def getOptions():
    usage = "Usage: python pg-pong [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize model, save to <model name>.pickle, then start training.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-s","--starting_ep", type="int", default=0, help="Starting episode number (for record keeping).");
    parser.add_option("-k","--k_steps", type="int", default=4, help="How many game steps to take before the model chooses a new action.");
    parser.add_option("-p","--play", action="store_true", default=False, help="Play only. No training and always choose the best action.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model, then exit.");
    parser.add_option("-g","--game", default="Breakout-v0", help="The game environment to use. Defaults to Breakout");
    parser.add_option("-m","--max_episodes", default="0", type="int", help="Maximum number of episodes to train.");
    parser.add_option("--upload", action="store_true", default=False, help="Monitor the training run and upload to OpenAI.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print( usage )
        exit()

    if options.k_steps != 1 and options.k_steps != 4:
        print( "Game step sizes other than 1 and 4 are not currently supported." )
        exit()

    if args[0].endswith('.hdf5'):
        args[0] = args[0][:-7]

    options.model_name = args[0]
    options.dir_model = os.path.expanduser(options.dir_model)

    return (options, args)

if __name__ == "__main__":
    options, _ = getOptions()

    env = gym.envs.make(options.game)
    nA = env.action_space.n
    if hasattr(env,'get_action_meanings'):
        print( env.get_action_meanings() )
    else:
        print( "Number of actions: {}".format( nA ) )

    if options.initialize:
        filename = os.path.join( options.dir_model, options.model_name + ".hdf5" )
        if os.path.exists(filename):
            print( "Model already exists at " + filename )
            print( "Delete the existing file or don't use the --initialize/-i flag." )
            exit()
        print( "Initializing model with %d actions..." % (nA,) )
        model = initializeModel( options.model_name, nA )
        model.env = options.game
        save_model( model, options.dir_model, options.model_name )
    else:
        print( "Reading model..." )
        model = read_model( options.dir_model, options.model_name )
        model.name = options.model_name
        model.env = options.game
        if not hasattr(model, 'env'):
            print( "Warning, model may not work with the current environment." )
      
    if options.desc:
        print( "Model: {}".format( model.name) )
        print( "Env  : {}".format( model.env ) )
        model.summary()
        exit()

    if options.upload:
        filename = os.path.join( options.dir_model, options.model_name + "_monitor" )
        env = gym.wrappers.Monitor(env, filename, force=True)

    if options.test_only:
        reward = test(model, env, options)
        print( "Average test reward: %f" % (reward,) )
        exit()

    if hasattr(model, 'env'):
        if model.env != options.game:
            print( "Model was not initialized for the current environment: %s vs %s" % (model.env,options.game) )
            exit()

    train(model, env, options)

    if options.upload:
        #gym.upload('./cartpole', api_key="")
        pass
