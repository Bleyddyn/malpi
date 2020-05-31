""" An OpenAI gym environment that uses a WorldModel trained on DonkeyCar data as the environment.

    To pass in the arguments needed during environment creation:
        your_env = gym.make('DKWMEnv', vae=vae_object, ...etc...)
        From: https://stackoverflow.com/questions/54259338/how-to-pass-arguments-to-openai-gym-environments-upon-init

    See this for how to create new environments: https://github.com/openai/gym/blob/master/docs/creating-environments.md
"""

from time import sleep

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

def shuffled_circular(data, default=None):
    while True:
        if data is None:
            yield default
        else:
            np.random.shuffle(data)
            for sample in data:
                yield sample

class DKWMRendererBase(object):

    def __init__( self, window_width=None, window_height=None):
        pass

    def reset(self):
        pass

    def set_obs( self, next_obs ):
        pass

    def clear_label( self, label_id ):
        pass

    def set_label( self, label_text, label_id, location=None ):
        pass

    def render( self, mode=None ):
        pass

    def close(self):
        pass

class DKWMRewardBase(object):

    def __init__(self, reward_range=(-10.0, 10.0)):
        self.reward_range = reward_range

    def reset(self):
        pass

    def step( self, z_obs=None, mu=None, var=None, obs=None, actions=None ):
        raise NotImplementedError

    def close(self):
        pass

class DKWMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, vae, rnn, reward_func, starts=None, renderer=None, max_steps=None):
        """ @param starts A list of mu/log_var pairs that can be sampled from to generate the first observation of each episode.
            @param vae A Variational Autoencoder with z_dim(), encode() and decode() methods.
            @param rnn An MDRNN  with hidden_units(), sample_next_output() and sample_z() methods.
            @param max_steps Will return done=True after max_steps
        """
        super().__init__()

        self.z_dim = vae.get_z_dim()
        self.vae = vae
        self.rnn = rnn
        self.max_steps = max_steps
        self.steps = 0

        if starts is not None:
            self.starts = shuffled_circular(starts, default=np.zeros( self.z_dim ))
        else:
            self.starts = None


        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (2,),
            dtype = np.float32
        )

        # z/latent space. Arbitrarily chosen min/max, should be based off normal distribution.
        self.observation_space = spaces.Box(
            low = -100.0,
            high = 100.0,
            shape = (self.z_dim,),
            dtype = np.float32
        )

        self.reward_func = reward_func
        if reward_func is None:
            self.reward_range = (-10.0,10.0)
        else:
            self.reward_range = self.reward_func.reward_range

        if renderer is None:
            self.renderer = DKWMRendererBase()
        else:
            self.renderer = renderer

    def reset(self):
        if self.starts is None:
            self.zobs = np.zeros( self.z_dim )
        else:
            self.zobs = self.rnn.sample_z( *next(self.starts) )

        self.hidden = np.zeros(self.rnn.get_hidden_units())
        self.cell_values = np.zeros(self.rnn.get_hidden_units())
        self.reward = 0.0
        self.steps = 0

        next_obs = self.zobs_to_obs( self.zobs )
        self.renderer.set_obs( next_obs )

        self.renderer.reset()

        return self.zobs

    def step(self, action):
        inputs = np.concatenate([self.zobs, action, [self.reward]])

        ret = self.rnn.sample_next_output(inputs, self.hidden, self.cell_values)
        self.zobs, mu, log_var, _, rew_pred, self.reward, self.hidden, self.cell_values = ret

        next_obs = self.zobs_to_obs( self.zobs )

        if self.reward_func is None:
            self.reward = np.clip( rew_pred, self.reward_range[0], self.reward_range[1] )
        else:
            self.reward = self.reward_func.step( z_obs=self.zobs, mu=mu, var=log_var, obs=next_obs, actions=action )

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        else:
            done = False

        self.renderer.set_obs( next_obs )
        self.renderer.set_label( "Steering:\t{:+5.3f}\nThrottle:\t{:+5.3f}".format( *action ), "actions"  )

        return self.zobs, self.reward, done, {"decoded": next_obs}

    def render(self, mode='human'):
        img = self.renderer.render( mode=mode )
        if "rgb_array" == mode:
            return img
        #if "human" == mode:
        #    sleep(0.13)

    def close(self):
        self.reward_func.close()
        self.renderer.close()

    def zobs_to_obs(self, zobs):
        next_obs = np.squeeze( self.vae.decode(zobs.reshape( (1,self.z_dim) )) ) * 255.0
        next_obs = np.floor( next_obs )
        next_obs = np.clip( next_obs, 0, 255 )
        return next_obs.astype( np.uint8 )

def sample_code():
    #import gym
    #import malpi.dkwm.gym_envs
    #from malpi.dkwm.gym_envs.lane_reward import LaneReward
    #from malpi.dkwm.gym_envs.renderer import DKWMRenderer
    #from malpi.dkwm import vae
    #from malpi.dkwm import mdrnn
    #import numpy as np
    #from gym import spaces

    print( "Gym: {} at {}".format( gym.__version__,  gym.__file__ ) )

    z_dim = 512
    vae = vae.KerasVAE(z_dim=z_dim, dropout=None)
    vae.set_weights( vae_path, by_name=True )
    rnn = mdrnn.RNN(z_dim=z_dim, action_dim=2 )
    rnn.set_weights( mdrnn_path, by_name=True )
    rew = LaneReward( z_dim=z_dim, weights=vae_path, reward_range=(-10.0, 10.0) )
    starts = np.load( starts_path )
    starts = list(zip(starts['mu'], starts['log_var']))
    obs_size=128
    renderer = DKWMRenderer( window_width=obs_size*2, window_height=obs_size*2 )

    # Passing arguments like this requires OpenAI gym >= 0.12.4
    env = gym.make('dkwm-v0', vae=vae, rnn=rnn, reward_func=rew, starts=starts, renderer=renderer)
    print( "Env: {}".format( env ) )

    obs = env.reset()
    print( "Obs: {}".format( obs.shape ) )

    for i in range(10):
        act = env.action_space.sample()
        z_obs, reward, done, info = env.step( env.action_space.sample() )
        obs = info["decoded"]
        print( "Step act/obs/rew/done/z: {} {} {} {} {}".format( act, obs.shape, reward, done, z_obs.shape ) )

# Sample output:
# Step act/obs/rew/done/z: [ 0.3011232  -0.97818303] (128, 128, 3) 0 False (512,)

    env.close()
