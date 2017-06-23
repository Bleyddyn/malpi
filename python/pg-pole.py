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

    with open('CartPole-v0_won.txt', 'r') as f:
        for line in f:
            resd = ast.literal_eval(line)
            if isinstance(resd,dict):
                best = 195.0
                if 'best_score' in resd:
                    best = resd['best_score']
                sample = [32, 10, 200, 0.99, resd['epsilon'], resd['epsilon_decay'],resd['learning_rate'],resd['learning_rate_decay'],resd['lr_decay_on_best'],resd['clip_error'], 0.005]
            elif isinstance(resd,list):
                sample = resd[0:10]
                sample.append(0.005)
                best = resd[10]
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

def softmax_batch(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

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

# batch_size update_rate update_freq gamma epsilon epsilon_decay learning_rate learning_rate_decay lr_decay_on_best clip_error behavior.reg

    next_sample = np.array( [ 32, 20, 100, 0.99, 0.9, 0.9995, 0.005, 0.9999, 0.95,True, 0.0005 ] )
#    ns2 = [35.471222966270744, 38.37807565265633, 116.15169967184646, 0.7994517015140111, 0.5467350837104111, 0.9931695064926428, 0.009819376267803895, 0.9967592218595942, 0.9663844877653254, 0.7082498175370553, 0.0020246883852151417]
#    ns3 = [15.794413185088402, 27.943625798031828, 798.1561128587946, 0.9542275528280187, 0.7105140406717579, 0.9996216020143134, 0.021327395517623794, 0.9996498782205984, 0.9583684951507172, 0.21068863599082088, 0.004261505455968546]
#    ns5 = [48.815653567290425, 34.961648567661825, 468.3846881487566, 0.23313941479454803, 0.1630527266282271, 0.9932152896891062, 0.0688362208079374, 0.9985657914516108, 0.9745687054036098, 0.7234555328172226, 0.004001796434991941]
#    ns6 = [28.24000380576149, 4.503855398537693, 647.7616508987576, 0.5136221792299456, 0.4310535569147862, 0.9921263218184515, 0.04364309633846753, 0.9968090204151728, 0.9815313824481013, 0.8650881828450184, 0.00560198882477674]
#    ns7 = [30.925956532141644, 21.645822197961028, 782.7258088986783, 0.9975081589468211, 0.5755960192901446, 0.9917304919341033, 0.09969599669488056, 0.9992139877010152, 0.947164407569207, 0.6338001376910157, 0.009939094019751054]
#    ns8 = [24.20062352160077, 31.63370169555912, 141.8076862504255, 0.6105570419507371, 0.4056939760149664, 0.9932989781711511, 0.0802181271288588, 0.9989581236209448, 0.9128658066048594, 0.7608427670235947, 0.0016435174719399933]
#    ns4 = [18.32026345019517, 45.64960707155015, 781.1920097253865, 0.12244453901068054, 0.2941830570247511, 0.9949184958539329, 0.01666072047036751, 0.9999725890071582, 0.9068317107623877, 0.4337409896399025, 0.003750798870686474]
#    ns9 = [10.25430465535929, 35.284676962320155, 595.7011299729893, 0.25599137210178063, 0.3280938239975178, 0.992898000862435, 0.02941715637109388, 0.9996840142279082, 0.926579984522795, 0.01586549543950433, 0.0048595528178426595]
    ns2 = [35.47122, 38.37808, 116.1517, 0.79945, 0.54674, 0.99317, 0.00982, 0.99676, 0.96638, 0.70825, 0.00202]
    ns3 = [15.79441, 27.94363, 798.15611, 0.95423, 0.71051, 0.99962, 0.02133, 0.99965, 0.95837, 0.21069, 0.00426]
    ns4 = [18.32026, 45.64961, 781.19201, 0.12244, 0.29418, 0.99492, 0.01666, 0.99997, 0.90683, 0.43374, 0.00375]
    ns5 = [48.81565, 34.96165, 468.38469, 0.23314, 0.16305, 0.99322, 0.06884, 0.99857, 0.97457, 0.72346, 0.004]
    ns6 = [28.24, 4.50386, 647.76165, 0.51362, 0.43105, 0.99213, 0.04364, 0.99681, 0.98153, 0.86509, 0.0056]
    ns7 = [30.92596, 21.64582, 782.72581, 0.99751, 0.5756, 0.99173, 0.0997, 0.99921, 0.94716, 0.6338, 0.00994]
    ns8 = [24.20062, 31.6337, 141.80769, 0.61056, 0.40569, 0.9933, 0.08022, 0.99896, 0.91287, 0.76084, 0.00164]
    ns9 = [10.2543, 35.28468, 595.70113, 0.25599, 0.32809, 0.9929, 0.02942, 0.99968, 0.92658, 0.01587, 0.00486]
    ns10= [41.21059904, 38.28170401, 894.50800476, 0.74548384, 0.55663432, 0.99318955, 0.07205227, 0.99999037, 0.93612528, 0.0950218, 0.004791, 200., 4598.]
    ns11= [19.92790589, 30.41193725, 143.25020338, 0.589395, 0.90056903, 0.99319374, 0.0621287, 0.99970608, 0.90470446, 0.26470385, 0.0078707, 200., 690.]
    ns12= [35.67269471, 39.23544342, 237.63394761, 0.49218579, 0.21019492, 0.99898196, 0.07409025, 0.99972359, 0.97938121, 0.29803843, 0.00675115, 200., 1287.]
    ns13= [15.13436928, 27.73185857, 173.97569991, 0.72684987, 0.72425576, 0.99975706, 0.0601372, 0.99826244, 0.98179968, 0.43993769, 0.00271678, 200., 972.]

    #x_list,y_list = readParams()
    x_list = [ns2,ns3,ns4,ns5,ns6,ns7,ns8,ns9,ns10[0:11],ns11[0:11],ns12[0:11]]
    y_list = [136.04,123.39,192.15, 138.44,62.33,128.06,189.53,192.88,ns10[11],ns11[11],ns12[11]]
    xp = np.array(x_list)
    yp = np.array(y_list)
    bounds = np.array( [ [10, 50], [1,50], [100,1000], [0.1,1.0], [0.1,1.0], [0.99,1.0], [0.0001,0.1], [0.99,1.0], [0.9,1.0],[0.0,1.0], [0.0005,0.01] ] )
    do_bayes = False
    scores = []

    if not do_bayes:
        next_sample = ns11
# Try setting learning rate decay to 1.0 so that RMSProp does all of the learning decay.
# Doesn't seem to work as well, so keep the learning rate decay
#        next_sample[7] = 1.0
#        next_sample[8] = 1.0

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

    behavior = initializeModel( options.model_name, num_actions, input_dim=(4,1) )
    behavior.reg = hparams[10]
    behavior.params["W1"] *= 0.1
    optim = Optimizer( "rmsprop", behavior, learning_rate=learning_rate, decay_rate=0.99, upd_frequency=update_freq)

    reward_sum = 0
    reward_100 = deque(maxlen=100)
    best_test = 15.0 # test(behavior, env, options)
    steps = 0
    episode_steps = 0
    episode_number = 0

    state = env.reset()
    exp_history = Experience2( 2000, state.shape )

    with open( os.path.join( options.game + ".txt" ), 'a+') as f:
        f.write( "%s = %s\n" % ('Start',time.strftime("%Y-%m-%d %H:%M:%S")) )
        f.write( "%s = %s\n" % ('Model Name',behavior.name) )
        if options.initialize:
            f.write( "Weights initialized\n" )
            f.write( str(behavior.layers) + "\n" )
            f.write( str(behavior.layer_params) + "\n" )
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

        actions_raw, _ = behavior.forward( state.reshape(1,4), mode="test")
        action_probs = softmax_batch(actions_raw)
        action = np.random.choice(num_actions, p=action_probs[0])
# Random action, for baseline scores
#        action = np.random.randint(num_actions)

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
  
        if done: # an episode finished
            states, actions, rewards, batch_done, new_states = exp_history.all()
  
            actions = actions.astype(np.int)
            rewards = discount_rewards( rewards, gamma, batch_done, normalize=True )
  
            actions_raw, caches = behavior.forward( states )
            action_probs = softmax_batch(actions_raw)

            gradients = action_probs[range(action_probs.shape[0]),actions]
            #print action_probs
            #print actions
            #print "action_probs: %s" % (action_probs.shape,)
            #print "Gradients: %s" % (gradients.shape,)
            #print "Rewards: %s" % (rewards.shape,)
            #dx = rewards / gradients
            #y = np.zeros(action_probs.shape)
            #y[range(action_probs.shape[0]),actions] = dx
            #dx = -y

            # From: https://github.com/keon/policy-gradient/blob/master/pg.py
            y = np.zeros(action_probs.shape)
            y[range(action_probs.shape[0]),actions] = 1.0
            gradients = y - action_probs
            gradients *= np.reshape(rewards, [rewards.shape[0],1])
            dx = -gradients

# Gradient of the action taken divided by the probability of the action taken
            #action_probs = np.zeros(action_probs.shape)
            #action_probs[actions] = 1.0

            # From: http://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html 
            # ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, ps)) # prevent log of zero
            #policy_grad_loss = -np.sum(np.log(ps) * actions_one_hot * advs)
# would still need the derivitive of the loss

            #dx = actions_raw + gradients
            #dx = -gradients
            loss = 0.0
             
            if clip_error:
                np.clip( dx, -1.0, 1.0, dx )

            # dx needs to have shape(batch_size,num_actions), e.g. (32,6)
            _, grad = behavior.backward(caches, loss, dx )
            optim.update( grad, check_ratio=True )

            episode_number += 1

            reward_100.append(reward_sum)

            exp_history.clear()

            if episode_number % update_rate == 0:
                treward = np.mean(reward_100) # test(behavior, env, options)

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
            if epsilon > 0.001:
                epsilon *= epsilon_decay
            reward_sum = 0
            episode_steps = 0
            steps = 0
            state = env.reset()

    with open( os.path.join( options.game + "_won.txt" ), 'a+') as f:
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

