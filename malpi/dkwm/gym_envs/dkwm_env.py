""" An OpenAI gym environment that uses a WorldModel trained on DonkeyCar data
    to create the environment.

    To pass in the arguments needed during environment creation:
        your_env = gym.make('DKWMEnv', z_dim=128, vae_weights="path/to/weights.h5', ...etc...)
        From: https://stackoverflow.com/questions/54259338/how-to-pass-arguments-to-openai-gym-environments-upon-init

    See this for how to create new environments: https://github.com/openai/gym/blob/master/docs/creating-environments.md
"""

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from malpi.dkwm import vae
from malpi.dkwm import mdrnn


class DKWMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, z_dim=128, vae_weights=None, rnn_weights=None, obs_height=128, obs_width=128):

        self.vae_weights = vae_weights
        self.rnn_weights = rnn_weights
        self.z_dim = z_dim

        # z_dim, dropout, aux = vae.KerasVAE.model_meta( vae_weights[:-3] + ".json" )
        self.vae = vae.KerasVAE(z_dim=z_dim, dropout=None)
        self.vae.set_weights( vae_weights )

        self.rnn = mdrnn.RNN(z_dim=z_dim, action_dim=2 )
        self.rnn.set_weights( rnn_weights )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2),
            dtype=np.float
        )

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (-10.0, 10.0)

    def reset(self):
        self.hidden = np.zeros(rnn.hidden_units)
        self.cell_values = np.zeros(rnn.hidden_units)
        self.zobs = np.zeros( self.z_dim )
        self.reward = 0.0

        next_obs = self.zobs_to_obs( self.zobs )

        return next_obs

    def step(self, action):
        inputs = np.concatenate([self.zobs, action, [self.reward]])

        ret = rnn.sample_next_output(inputs, self.hidden, self.cell_values)
        self.zobs, _, _, _, rew_pred, self.reward, self.hidden, self.cell_values = ret
        
        next_obs = self.zobs_to_obs( self.zobs )
        done = False

        return next_obs, self.reward, done, {"z_obs": self.zobs}

    def render(self, mode='human'):
        # Possible code to base it on: https://github.com/maximecb/gym-miniworld/blob/master/gym_miniworld/miniworld.py
        pass

    def close(self):
        pass

    def zobs_to_obs(self, zobs):
        next_obs = np.squeeze( self.vae.decode(zobs.reshape( (1,self.z_dim) )) ) * 255.0
        next_obs = np.floor( next_obs )
        next_obs = np.clip( next_obs, 0, 255 )
        return next_obs.astype( np.uint8 )
