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

    with open('CartPole-v0_dqn_won.txt', 'r') as f:
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
    layers = ["FC-200", output]
    layer_params = [ {}, {'relu':False} ]
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
    good_list = [ [48.76090415229852, 1.6927175081103532, 223.02341007066963, 0.5678010007909667, 0.5549954432648416, 0.9933886373603302, 0.04461187669276121, 0.9911454128640629, 0.9563065642076264, 0.8080555822008355, 0.0015395898545990808, 165.0, 5000.0],
                [44.05077224914717, 4.581278696929539, 567.454951740726, 0.872342953950116, 0.18049877148475657, 0.990163709408304, 0.062370916712252866, 0.9944033072903318, 0.9963371399648688, 0.6395886294825446, 0.0010001848382758618, 117.75, 5000.0],
                [36.3723209296144, 17.13540920146732, 649.2612028561178, 0.13224300863461783, 0.543166266140875, 0.9943310250757145, 0.08187568538373177, 0.9966047176258499, 0.996227517495977, 0.4472571272004753, 0.00832929196043553, 105.12, 5000.0],
                [10.916692091321929, 1.7197588754360758, 859.5984930832783, 0.9928960793644326, 0.1274628002990129, 0.9905321890913572, 0.08505446936131436, 0.9954039819492306, 0.9393970414024277, 0.20165955117569845, 0.00393562696555546, 184.0, 5000.0],
                [24.61024966623437, 2.3382317127384797, 125.6807628925593, 0.7513928228888437, 0.2758971455651426, 0.9928318138327047, 0.013857939559698086, 0.9927166247992542, 0.9609541841323385, 0.4939770517123132, 0.004033141328968626, 127.14, 5000.0],
                [48.414647941793945, 29.60459215462402, 929.5659155100193, 0.22797686540871967, 0.29012857317101626, 0.9902589981938963, 0.048126323473176816, 0.999365668290878, 0.9537590730846931, 0.3837955994859634, 0.0046700024476340925, 131.60344827586206, 5000.0],
                [11.625857336308801, 1.7992254729400174, 834.250910881173, 0.9904487770340547, 0.1441466452323528, 0.99, 0.08112103123697603, 0.9967248247150776, 0.9628560158758284, 0.64953096598099, 0.005206558865528496, 134.0, 5000.0],
                [ 32, 20, 100, 0.99, 0.7, 0.9995, 0.01, 0.9999, 0.95,True, 0.0005, 195.0, 2000.0 ]
                ]
    x_list = []
    y_list = []
    for param in good_list:
        x_list.append( param[0:11] )
        y_list.append( param[11] )
    xp = np.array(x_list)
    yp = np.array(y_list)
# batch_size update_rate update_freq gamma epsilon epsilon_decay learning_rate learning_rate_decay lr_decay_on_best clip_error behavior.reg
    bounds = np.array( [ [10, 50], [1,50], [100,1000], [0.1,1.0], [0.1,1.0], [0.99,1.0], [0.0001,0.1], [0.99,1.0], [0.9,1.0],[0.0,1.0], [0.0005,0.01] ] )
    do_bayes = False
    do_uniform = False
    do_normal = False
    next_sample = np.array( [ 32, 20, 100, 0.99, 0.7, 0.9995, 0.01, 0.9999, 0.95,True, 0.0005 ] )
    scores = []

    for i in range(100):
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
                stddev = [ 5.0, 0.1, 50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.5, 0.001 ]
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

    batch_size = int(hparams[0])
    update_rate = int(hparams[1])
    update_freq = int(hparams[2])
    gamma = hparams[3]
    epsilon = hparams[4]
    epsilon_decay = hparams[5]
    learning_rate = hparams[6]
    learning_rate_decay = hparams[7]
    lr_decay_on_best = hparams[8]
    if hparams[9] < 0.5:
        clip_error = False
    else:
        clip_error = True

    target = initializeModel( options.model_name, num_actions, input_dim=(4,1) )
    target.reg = hparams[10]
    target.params["W1"] *= 0.1
    behavior = copy.deepcopy(target)
    optim = Optimizer( "rmsprop", behavior, learning_rate=learning_rate, decay_rate=0.99, upd_frequency=update_freq)

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
        f.write( "%s = %s\n" % ('Model Name',target.name) )
        if options.initialize:
            f.write( "Weights initialized\n" )
            f.write( str(target.layers) + "\n" )
            f.write( str(target.layer_params) + "\n" )
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
        f.write( "Optimizer %s\n" % (optim.optim_type,) )
        f.write( "   %s = %f\n" % ('learning rate',optim.learning_rate) )
        f.write( "   %s = %f\n" % ('decay rate',optim.decay_rate) )
        f.write( "   %s = %f\n" % ('epsilon',optim.epsilon) )
        f.write( "   %s = %f\n" % ('update frequency',optim.upd_frequency) )
        f.write( "\n" )

    while (options.max_episodes == 0) or (episode_number < options.max_episodes):
      if options.render: env.render()

      action = choose_epsilon_greedy( behavior, state, epsilon, num_actions )
      #action = np.random.randint(num_actions)

      # step the environment once, or ksteps times
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

      exp_history.save( state, action, reward, done, next_state )
      state = next_state

      if (exp_history.size() > (batch_size * 5)):
          if ( len(exp_history.priority) < 1) or (np.random.uniform(0,10) < 9):
              states, actions, rewards, batch_done, new_states = exp_history.batch( batch_size )
          else:
              states, actions, rewards, batch_done, new_states = exp_history.priority_batch()
              print "Priority Batch"

          actions = actions.astype(np.int)

          target_values, _ = target.forward( new_states, mode='test' )

          double_dqn = True
          if double_dqn:
              behavior_values, _ = behavior.forward( new_states, mode='test' )
              best_actions = np.argmax(behavior_values,axis=1)
              q_target = rewards + batch_done * gamma * target_values[np.arange(batch_size), best_actions]
          else:
              q_target = rewards + batch_done * gamma * np.max(target_values, axis=1)

          action_values, cache = behavior.forward(states, mode='train', verbose=False)

          q_error = np.zeros( action_values.shape )
          #q_error[ np.arange(batch_size), actions ] = q_target - action_values[ np.arange(batch_size), actions ]
          q_error[ np.arange(batch_size), actions ] = action_values[ np.arange(batch_size), actions ] - q_target
          dx = q_error
          dx /= batch_size
          if clip_error:
              np.clip( dx, -1.0, 1.0, dx )

          q_error = np.sum( np.square( q_error ) )

          # dx needs to have shape(batch_size,num_actions), e.g. (32,6)
          _, grad = behavior.backward(cache, q_error, dx )
          optim.update( grad, check_ratio=False )

      if done: # an episode finished
        episode_number += 1

        reward_100.append(reward_sum)

        if episode_number % update_rate == 0:

            target = copy.deepcopy(behavior)

            treward = np.mean(reward_100) # test(target, env, options)

            print
            print 'Ep %d' % ( episode_number, )
            print 'Reward       : %0.2f  %0.2f' % ( reward_sum, np.mean(reward_100) )
            print "Test reward  : %0.2f vs %0.2f" % (treward, best_test)
            print "Learning rate: %g" % (optim.learning_rate,)
            print "Epsilon      : %g" % (epsilon,)

            if treward > best_test:
                best_test = treward

                if treward > 195.0:
                    print "Final Learning rate: %f" % (optim.learning_rate,)
                    print "WON! In %d episodes" % (episode_number,)
                    break

                if optim.learning_rate > 0.00001:
                    optim.learning_rate *= lr_decay_on_best

        if optim.learning_rate > 0.00001:
            optim.learning_rate *= learning_rate_decay
        if epsilon > 0.1:
            epsilon *= epsilon_decay
        reward_sum = 0
        episode_steps = 0
        steps = 0
        state = env.reset()

    with open( os.path.join( options.game + "_dqn_won.txt" ), 'a+') as f:
        hparams = np.append( hparams, [best_test, episode_number] )
        f.write( "%s\n" % (hparams.tolist(),) )

    with open( os.path.join( options.game + ".txt" ), 'a+') as f:
        f.write( "%s = %f\n" % ('Final epsilon', epsilon) )
        f.write( "%s = %f\n" % ('Final learning rate', optim.learning_rate) )
        f.write( "%s = %f\n" % ('Best test score', best_test) )
        f.write( "%s = %d\n" % ('Episodes', episode_number) )
        f.write( "\n\n" )

    return best_test


def getOptions():
    usage = "Usage: python pg-pong [options] <model name>"
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

    options.model_name = "HyperParamSearch"

    if options.desc or options.test_only:
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

