""" Trains an agent with Actor/Critic on CartPole. Uses OpenAI Gym.
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
from malpi.experience import Experience2

try:
    import config
except:
    print "Failed to load config file config.py."
    print "Try copying config_empty.py to config.py and re-running."
    exit()

import ast
from sklearn.linear_model import BayesianRidge, LinearRegression
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp
# {'epsilon_decay': 0.99957392597900963, 'epsilon': 0.96126118058910504, 'learning_rate': 0.0048160891703121133, 'batch_size': 32, 'best_score': 164.90000000000001, 'episodes': 3000, 'clip_error': False, 'learning_rate_decay': 0.99992369857077323, 'lr_decay_on_best': 0.94999999999999996, 'update_rate': 20, 'reg': 0.0050000000000000001, 'gamma': 0.99}

def readParams():
    hparams = []
    y = []

    with open('CartPole-v0_ac_won.txt', 'r') as f:
        for line in f:
            resd = ast.literal_eval(line)
            if isinstance(resd,dict):
                best = 195.0
                if 'best_score' in resd:
                    best = resd['best_score']
                sample = [32, 10, 200, 0.99, resd['epsilon'], resd['epsilon_decay'],resd['learning_rate'],resd['learning_rate_decay'],resd['lr_decay_on_best'],resd['clip_error'], 0.005]
            elif isinstance(resd,list):
                sample = resd[0:11]
                best = resd[11]
            hparams.append(sample)
            y.append(best)

    #hparams = np.array(hparams)
    #y = np.array(y)
    return (hparams,y)


#clf = BayesianRidge(compute_score=True)
#clf.fit(hparams, y)

#ols = LinearRegression()
#ols.fit(X, y)

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
    print "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs)
            
def saveModel( model, options ):
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( model, f, pickle.HIGHEST_PROTOCOL)

def initializeModel( name, number_actions, input_dim=(4,84,84), verbose=False ):
    output = "FC-%d" % (number_actions,)
#    layers = ["conv-32", "maxpool", "conv-64", "maxpool", "conv-64", "FC-512", output]
#    layer_params = [{'filter_size':3, 'stride':1 },
#        {'pool_stride': 2, 'pool_width': 2, 'pool_height': 2},
#        {'filter_size':3, 'stride':1 },
#        {'pool_stride': 2, 'pool_width': 2, 'pool_height': 2},
#        {'filter_size':3, 'stride':2 },
#        {}, {'relu':False} ]
# From the DQN paper, mostly
#    layers = ["conv-32", "conv-64", "conv-64", "FC-512", output]
#    layer_params = [{'filter_size':8, 'stride':4, 'pad':4 },
#        {'filter_size':4, 'stride':2, 'pad':2},
#        {'filter_size':3, 'stride':1, 'pad':1},
#        {}, {'relu':False} ]
    layers = ["FC-20", "FC-20", output]
    layer_params = [ {}, {}, {'relu':False} ]
    model = MalpiModel(layers, layer_params, input_dim=input_dim, reg=0.005, dtype=np.float32, verbose=verbose)
    model.name = name

    if verbose:
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

def discount_rewards(r, gamma, done, normalize=True):
    """ take 1D float array of rewards and compute discounted reward.
        if normalize is True: subtract mean and divide by std dev
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if not done[t]: running_add = 0 # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    if normalize:
# standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

    return discounted_r

def softmax_batch(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

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
        q_values,_ = estimator.forward(observation.reshape(1,4), mode="test")
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

def hyperparameterGenerator( oneRun = False ):
    batch_size = 32 # backprop batch size
    update_rate = 20 # every how many episodes to copy behavior model to target
    gamma = 0.99 # discount factor for reward
    epsilon = 0.5
    epsilon_decay = 0.999
    learning_rate = 0.01
    learning_rate_decay = 0.999
    lr_decay_on_best = 0.95
    clip_error = True
    reg = 0.005

    hparams = { "reg": reg, "learning_rate": learning_rate, "learning_rate_decay":learning_rate_decay, "batch_size":batch_size, "update_rate":update_rate, "gamma":gamma, "epsilon":epsilon, "epsilon_decay":epsilon_decay,
        "lr_decay_on_best":lr_decay_on_best, "clip_error":clip_error }

    variations = np.array([0.9,1.0,1.1])
    if oneRun:
        reguls = [3.37091767808e-05]
        lrs = [0.0002006801544726]
    else:
        count = 4
        reguls = np.array([0.005])
        epsilons = np.random.uniform( 0.5, 1.0, count )
        epsilon_decays = np.random.uniform( 0.999, 0.9999, count )
        lrs = np.random.uniform( 0.0001, 0.03, count )
        lr_decays = np.random.uniform( 0.999, 0.99999, count )
        decays_on_best = np.array([lr_decay_on_best])
        clip_errors = np.array([True,False])
#        reguls = np.array([3.37091767808e-05]) * variations
#        lrs = np.array([0.0002006801544726]) * variations
#reguls = 10 ** np.random.uniform(-5, -4, 2) #[0.0001, 0.001, 0.01]
#lrs = 10 ** np.random.uniform(-6, -3, 5) #[1e-4, 1e-3, 1e-2]
#reguls = np.append([3.37091767808e-05],reguls)
#lrs = np.append([0.000182436504066],lrs)

    for reg in reguls:
        for lr in lrs:
            for decay in lr_decays:
                for epsilon in epsilons:
                    for epsilon_decay in epsilon_decays:
                        for decay_on_best in decays_on_best:
                            for clip_error in clip_errors:
                                hparams["reg"] = reg
                                hparams["learning_rate"] = lr
                                hparams["learning_rate_decay"] = decay
                                hparams["epsilon"] = epsilon
                                hparams["epsilon_decay"] = epsilon_decay
                                hparams["lr_decay_on_best"] = decay_on_best
                                hparams["clip_error"] = clip_error
                                yield hparams

def test(tmodel, env, options):
    reward_100 = 0
    for i in range(100):
        episode_reward = 0
        state = env.reset()
        done = False
        steps = 0
        while not done and (steps < 1000):
            if options.render: env.render()
            q_values,_ = tmodel.forward(state.reshape(1,4), mode="test")
            action = np.argmax(q_values[0])
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        reward_100 += episode_reward
    return (reward_100 / 100.0)

def train(env, options):
    alpha=1e-5
    epsilon=1e-7
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    #x_list,y_list = readParams()
    good_list = [
        [0.22136026231898764, 0.03435418156855076, 0.014258056717745918, 0.9932171398308102, 0.004031326729049222, 60.1, 5000.0],
        [0.8393075532448522, 0.05341277570972287, 0.03782414246524008, 0.9999393627274482, 0.006904643159300169, 63.28, 5000.0],
        [0.8165787628717937, 0.09427020466780384, 0.020845128577212314, 0.9992466981106534, 0.003885223951761664, 103.5, 5000.0],
        [0.47918425726181, 0.03429596907213825, 0.032978392062649034, 0.9915616675283389, 0.007389434048353167, 52.6875, 5000.0],
        [0.2449330437218338, 0.0947400634523504, 0.02914330566301235, 0.9908877396055589, 0.009001180405105197, 52.71, 5000.0],
        [0.7574053941509864, 0.022851358621574128, 0.02933848823742234, 0.9942378278140936, 0.0058316089556126535, 54.74, 5000.0],
        [0.5621275494548085, 0.07959943392424593, 0.06102889525003826, 0.9991853126845361, 0.005376755258427058, 65.56, 5000.0]
                ]
    x_list = []
    y_list = []
    for param in good_list:
        x_list.append( param[0:5] )
        y_list.append( param[5] )
    xp = np.array(x_list)
    yp = np.array(y_list)
# batch_size update_rate update_freq gamma epsilon epsilon_decay learning_rate-actor learning-rate-critic learning_rate_decay lr_decay_on_best clip_error behavior.reg
# gamma learning_rate-actor learning-rate-critic learning_rate_decay behavior.reg
#    bounds = np.array( [ [10, 50], [1,50], [100,1000], [0.1,1.0], [0.1,1.0], [0.99,1.0], [0.0001,0.1], [0.0001,0.1], [0.99,1.0], [0.9,1.0], [0.0,1.0], [0.0005,0.01] ] )
    bounds = np.array( [ [0.1,1.0], [0.0001,0.1], [0.0001,0.1], [0.99,1.0], [0.0005,0.01] ] )
    do_bayes = True
    do_uniform = False
    do_normal = False
    next_sample = np.array( [ 32, 20, 100, 0.99, 0.7, 0.9995, 0.01, 0.9999, 0.95,True, 0.0005 ] )
    scores = []

    for i in range(300):
        if do_bayes:
            model.fit(xp, yp)

            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            if np.any(np.abs(next_sample - xp) <= epsilon):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

            #next_sample = [32, 20, 200, 0.99, 0.88, 0.99957, 0.0045, 0.9999, 0.95, True, 0.005]
            # Sample loss for new set of parameters
            cv_score = train_one(env, next_sample, options)
            scores.append(cv_score)
            print "Score %f for %s" % (cv_score, next_sample)

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)
        else:
            if do_uniform:
                next_sample = []
                for b in bounds:
                    next_sample.append( np.random.uniform( b[0], b[1] ) )
            elif do_normal:
                next_sample = []
                stddev = [ 5.0, 0.1, 50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.5, 0.001 ]
                stdi = 0
                for b in good_list[3][0:11]:
                    next_sample.append( np.random.normal( b, stddev[stdi] ) )
                    stdi += 1
                bt = bounds.T
                next_sample = np.clip( next_sample, bt[0], bt[1] )
                print next_sample
            cv_score = train_one(env, next_sample, options)
            scores.append(cv_score)

    print "100 iterations: %f / %f" % (np.mean(scores), np.std(scores))

def train_one(env, hparams, options):
    ksteps = options.k_steps # number of frames to skip before selecting a new action
    num_actions = env.action_space.n

    batch_size = 32
    update_rate = 20
    update_freq = 500
    gamma = hparams[0]
    learning_rate_actor = hparams[1]
    learning_rate_critic = hparams[2]
    learning_rate_decay = hparams[3]
    clip_error = True
    off_line = False # Train on-line or off-line
    min_history = batch_size
    if off_line: min_history *= 5

    critic = initializeModel( options.model_name, 1, input_dim=(4,1) )
    actor = initializeModel( options.model_name, num_actions, input_dim=(4,1) )
    actor.reg = hparams[4]
    critic.reg = hparams[4]
    #actor.params["W1"] *= 0.1
    #critic.params["W1"] *= 0.1
    optim_critic = Optimizer( "rmsprop", critic, learning_rate=learning_rate_critic, decay_rate=0.99, upd_frequency=update_freq)
    optim_actor = Optimizer( "rmsprop", actor, learning_rate=learning_rate_actor, decay_rate=0.99, upd_frequency=update_freq)

    reward_sum = 0
    reward_100 = deque(maxlen=100)
    best_test = 15.0 # test(target, env, options)
    steps = 0
    episode_steps = 0
    episode_number = 0

    state = env.reset()
    exp_history = Experience2( 2000, state.shape )

    with open( os.path.join( options.game + ".txt" ), 'a+') as f:
        f.write( "%s = %s\n" % ('Start',time.strftime("%Y-%m-%d %H:%M:%S")) )
        f.write( "%s = %s\n" % ('Model Name',actor.name) )
        if options.initialize:
            f.write( "Weights initialized\n" )
            f.write( str(actor.layers) + "\n" )
            f.write( str(actor.layer_params) + "\n" )
        f.write( "%s = %d\n" % ('batch_size',batch_size) )
        f.write( "%s = %d\n" % ('update_rate',update_rate) )
        f.write( "%s = %f\n" % ('gamma',gamma) )
        f.write( "%s = %d\n" % ('k-steps',ksteps) )
        f.write( "%s = %f\n" % ('learning_rate_decay',learning_rate_decay) )
        f.write( "%s = %s\n" % ('clip_error',str(clip_error)) )
        f.write( "Optimizer Critic %s\n" % (optim_critic.optim_type,) )
        f.write( "   %s = %f\n" % ('learning rate',optim_critic.learning_rate) )
        f.write( "   %s = %f\n" % ('decay rate',optim_critic.decay_rate) )
        f.write( "   %s = %f\n" % ('update frequency',optim_critic.upd_frequency) )
        f.write( "Optimizer Actor %s\n" % (optim_actor.optim_type,) )
        f.write( "   %s = %f\n" % ('learning rate',optim_actor.learning_rate) )
        f.write( "   %s = %f\n" % ('decay rate',optim_actor.decay_rate) )
        f.write( "   %s = %f\n" % ('update frequency',optim_actor.upd_frequency) )
        f.write( "\n" )

    while (options.max_episodes == 0) or (episode_number < options.max_episodes):
      if options.render: env.render()

      actions_raw, _ = actor.forward( state.reshape(1,4), mode="test")
      action_probs = softmax_batch(actions_raw)
      action_probs = action_probs[0]
      action = np.random.choice(num_actions, p=action_probs)

      reward = 0
      done = False
      for k in range(ksteps):
          next_state, r, d, info = env.step(action)
          reward += r
          if d:
              done = True

      reward_sum += reward

      steps += ksteps
      episode_steps += ksteps

      exp_history.save( state, action, reward, done, next_state, action_probs )
      state = next_state

      if (exp_history.size() >= min_history):
          states, actions, rewards, batch_done, new_states, batch_probs = exp_history.batch( batch_size )

          actions = actions.astype(np.int)

          state_values, cache = critic.forward( states, mode='train', verbose=False )
          next_values, _ = critic.forward( new_states, mode='test' )

          td_error = np.reshape(rewards,(batch_size,1)) + (np.reshape(batch_done,(batch_size,1)) * ((gamma * next_values) - state_values))

          dx = td_error
          dx /= batch_size
          if clip_error:
              np.clip( dx, -5.0, 5.0, dx )

          q_error = 0.0

          _, grad = critic.backward(cache, q_error, dx )
          optim_critic.update( grad, check_ratio=False )

          actions_raw, acache = actor.forward( states, mode="train", verbose=False )
          action_probs = softmax_batch(actions_raw)
          #p = action_probs / batch_probs

          y = np.zeros(action_probs.shape)
          y[range(action_probs.shape[0]),actions] = 1.0
          gradients = y - action_probs
          gradients *= np.reshape(td_error, [td_error.shape[0],1])
          dx = -gradients

          #dx *= p
          dx /= batch_size
          if clip_error:
              np.clip( dx, -5.0, 5.0, dx )
          _, agrad = actor.backward(acache, q_error, dx )
          optim_actor.update( grad, check_ratio=False )

          if not off_line:
              exp_history.clear()

      if done: # an episode finished
        episode_number += 1

        reward_100.append(reward_sum)

        if episode_number % update_rate == 0:

            treward = np.mean(reward_100) # test(target, env, options)

            print
            print 'Ep %d' % ( episode_number, )
            print 'Reward       : %0.2f  %0.2f' % ( reward_sum, np.mean(reward_100) )
            print "Test reward  : %0.2f vs %0.2f" % (treward, best_test)
            print "Actor LR     : %g" % (optim_actor.learning_rate,)
            print "Critic LR    : %g" % (optim_critic.learning_rate,)

            if treward > best_test:
                best_test = treward

                if treward > 195.0:
                    print "Final Actor LR : %f" % (optim_actor.learning_rate,)
                    print "Final Critic LR: %f" % (optim_critic.learning_rate,)
                    print "WON! In %d episodes" % (episode_number,)
                    break

        if optim_critic.learning_rate > 0.00001:
            optim_critic.learning_rate *= learning_rate_decay
        if optim_actor.learning_rate > 0.00001:
            optim_actor.learning_rate *= learning_rate_decay
        reward_sum = 0
        episode_steps = 0
        steps = 0
        state = env.reset()

    with open( os.path.join( options.game + "_ac_won.txt" ), 'a+') as f:
        hparams = np.append( hparams, [best_test, episode_number] )
        f.write( "%s\n" % (hparams.tolist(),) )

    with open( os.path.join( options.game + ".txt" ), 'a+') as f:
        f.write( "%s = %f\n" % ('Final Actor LR', optim_actor.learning_rate) )
        f.write( "%s = %f\n" % ('Final Critic LR', optim_critic.learning_rate) )
        f.write( "%s = %f\n" % ('Best test score', best_test) )
        f.write( "%s = %d\n" % ('Episodes', episode_number) )
        f.write( "\n\n" )

    return best_test


def getOptions():
    usage = "Usage: python ac-pole.py [options] <model name>"
    parser = OptionParser( usage=usage )
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize model, save to <model name>.pickle, then start training.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment while training. Will greatly reduce speed.");
    parser.add_option("-s","--starting_ep", type="int", default=0, help="Starting episode number (for record keeping).");
    parser.add_option("-k","--k_steps", type="int", default=1, help="How many game steps to take before the model chooses a new action.");
    parser.add_option("-p","--play", action="store_true", default=False, help="Play only. No training and always choose the best action.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model, then exit.");
    parser.add_option("-g","--game", default="Breakout-v0", help="The game environment to use. Defaults to Breakout.");
    parser.add_option("-m","--max_episodes", default="0", type="int", help="Maximum number of episodes to train.");
    parser.add_option("--upload", action="store_true", default=False, help="Monitor the training run and upload to OpenAI.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        exit()

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]
    options.model_name = args[0]

    if options.k_steps != 1 and options.k_steps != 4:
        print "Game step sizes other than 1 and 4 are not currently supported."
        exit()

    options.dir_model = os.path.expanduser(options.dir_model)

    return (options, args)

if __name__ == "__main__":
    options, _ = getOptions()

    env = gym.envs.make(options.game)
    if hasattr(env,'get_action_meanings'):
        print env.get_action_meanings()

    if options.desc or options.test_only:
        if options.initialize:
            filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
            if os.path.exists(filename):
                print "Model already exists at " + filename
                print "Delete the existing file or don't use the --initialize/-i flag."
                exit()
            nA = env.action_space.n
            print "Initializing model with %d actions..." % (nA,)
            model = initializeModel( options.model_name, nA, input_dim=(4,1) )
            model.params["W1"] *= 0.1
            model.describe()
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
            if hasattr(model, 'env'):
                if model.env != options.game:
                    print "Model was not initialized for the current environment: %s vs %s" % (model.env,options.game)
                    exit()

            treward = test(model, env, options)
            print "Gym reward: %f" % treward
            exit()

    if options.upload:
        env = gym.wrappers.Monitor(env, "./" + options.game, force=True)

    train(env, options)

    env.close()

    if options.upload:
        if hasattr(config, 'openai_key'):
            gym.upload('./' + options.game, api_key=config.openai_key)
        else:
            print "Unable to upload results. Missing 'openai_key' in config."

