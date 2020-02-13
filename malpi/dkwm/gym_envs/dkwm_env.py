""" An OpenAI gym environment that uses a WorldModel trained on DonkeyCar data
    to create the environment.

    To pass in the arguments needed during environment creation:
        your_env = gym.make('DKWMEnv', z_dim=128, vae_weights="path/to/weights.h5', ...etc...)
        From: https://stackoverflow.com/questions/54259338/how-to-pass-arguments-to-openai-gym-environments-upon-init

    See this for how to create new environments: https://github.com/openai/gym/blob/master/docs/creating-environments.md
"""

from time import sleep

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from malpi.dkwm import vae
from malpi.dkwm import mdrnn
from malpi.dkwm.gym_envs.renderer import DKWMRenderer

def shuffled_circular(data, default=None):
    while True:
        if data is None:
            yield default
        else:
            np.random.shuffle(data)
            for sample in data:
                yield sample

class DKWMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, z_dim=128, vae_weights=None, rnn_weights=None, obs_height=128, obs_width=128, starts=None):
        """ @param starts A list of mu/log_var pairs that can be sampled from to generate the first observation of each episode.
        """
        self.vae = None
        self.rnn = None

        if starts is not None:
            self.starts = shuffled_circular(starts, default=np.zeros( z_dim ))
        else:
            self.starts = None

        self.load_weights( z_dim=z_dim, vae_weights=vae_weights, rnn_weights=rnn_weights )

        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (2,),
            dtype = np.float32
        )

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (-10.0, 10.0)

        self.renderer = DKWMRenderer( window_width=obs_width*2, window_height=obs_height*2 )

    def load_weights(self, z_dim, vae_weights, rnn_weights):
        self.vae_weights = vae_weights
        self.rnn_weights = rnn_weights
        self.z_dim = z_dim

        # z_dim, dropout, aux = vae.KerasVAE.model_meta( vae_weights[:-3] + ".json" )
        if self.vae_weights is not None:
            self.vae = vae.KerasVAE(z_dim=z_dim, dropout=None)
            self.vae.set_weights( vae_weights )

        if self.rnn_weights is not None:
            self.rnn = mdrnn.RNN(z_dim=z_dim, action_dim=2 )
            self.rnn.set_weights( rnn_weights )

    def reset(self):
        if self.vae_weights is None or self.rnn_weights is None:
            print( "Warning! Weights files have not been set for this DKWM environement." )
            print( "   Call env.load_weights( z_dim, vae_weights, rnn_weights ) before trying to run." )
            return None

        if self.starts is None:
            self.zobs = np.zeros( self.z_dim )
        else:
            self.zobs = mdrnn.RNN.sample_z( *next(self.starts) )

        self.hidden = np.zeros(self.rnn.hidden_units)
        self.cell_values = np.zeros(self.rnn.hidden_units)
        self.reward = 0.0

        next_obs = self.zobs_to_obs( self.zobs )

        self.renderer.reset()

        return next_obs

    def step(self, action):
        inputs = np.concatenate([self.zobs, action, [self.reward]])

        ret = self.rnn.sample_next_output(inputs, self.hidden, self.cell_values)
        self.zobs, _, _, _, rew_pred, self.reward, self.hidden, self.cell_values = ret
        
        next_obs = self.zobs_to_obs( self.zobs )
        done = False

        self.renderer.set_obs( next_obs )
        self.renderer.set_label( "Steering:\t{:+5.3f}\nThrottle:\t{:+5.3f}".format( *action ), "actions"  )

        return next_obs, self.reward, done, {"z_obs": self.zobs}

    def render(self, mode='human'):
        # Possible code to base it on: https://github.com/maximecb/gym-miniworld/blob/master/gym_miniworld/miniworld.py
        img = self.renderer.render( mode=mode )
        if "rgb_array" == mode:
            return img
        #if "human" == mode:
        #    sleep(0.13)

    def close(self):
        self.renderer.close()

    def zobs_to_obs(self, zobs):
        next_obs = np.squeeze( self.vae.decode(zobs.reshape( (1,self.z_dim) )) ) * 255.0
        next_obs = np.floor( next_obs )
        next_obs = np.clip( next_obs, 0, 255 )
        return next_obs.astype( np.uint8 )

def sample_code():
    #import gym
    #import malpi.dkwm.gym_envs
    #import numpy as np
    #from gym import spaces

    print( "Gym: {} at {}".format( gym.__version__,  gym.__file__ ) )

    # Passing arguments like this requires OpenAI gym >= 0.12.4
    env = gym.make('dkwm-v0', z_dim=512, vae_weights="vae_model.h5", rnn_weights="mdrnn_model.h5")
    print( "Env: {}".format( env ) )

    obs = env.reset()
    print( "Obs: {}".format( obs.shape ) )

    for i in range(10):
        act = env.action_space.sample()
        obs, reward, done, info = env.step( env.action_space.sample() )
        z_obs = info["z_obs"]
        print( "Step act/obs/rew/done/z: {} {} {} {} {}".format( act, obs.shape, reward, done, z_obs.shape ) )

# Sample output:
# Step act/obs/rew/done/z: [ 0.3011232  -0.97818303] (128, 128, 3) 0 False (512,)

    env.close()
