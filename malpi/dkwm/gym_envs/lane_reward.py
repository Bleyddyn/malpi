import numpy as np
from malpi.dkwm.gym_envs.dkwm_env import DKWMRewardBase
from keras.layers import Input, Dense
from keras.models import Model

class LaneReward(DKWMRewardBase):
    """ Lane definitions: ["InsideTrack", "InsideLine", "InsideLane", "MiddleLine", "OutsideLane", "OutsideLine", "OutsideTrack"]
    """
    def __init__(self, z_dim, weights, reward_range=(-10.0, 10.0)):
        super().__init__(reward_range=reward_range)
        self.z_dim = z_dim
        self.reward_scales = [-10.0, -10.0, 2.0, 1.0, 0.5, -10.0, -10.0]
        self.aux = len(self.reward_scales)
        self.model = self.make_model()
        self.load_weights(weights, by_name=True)

    def load_weights(self, weights, by_name=False):
        self.weights = weights
        if self.weights is not None:
            self.model.set_weights( weights, by_name=by_name )

    def reset(self):
        pass

    def step( self, z_obs=None, mu=None, var=None, obs=None, actions=None ):
        lane_cat = np.squeeze( self.model.predict( z_obs.reshape( (1,self.z_dim) )) )
        lane = np.argmax(lane_cat)
        # Multiply the reward scale for the current lane times the speed.
        reward = self.reward_scales[lane] * actions[1]
        reward = np.clip( reward, self.reward_range[0], self.reward_range[1] )
        return reward

    def close(self):
        pass

    def make_model(self):
        z_input = Input(shape=(self.z_dim,), name='z_input')
        aux_dense1 = Dense(100, name="aux1")(z_input)
        aux_dense2 = Dense(50, name="aux2")(aux_dense1)
        aux_out = Dense(self.aux, name="aux_output", activation='softmax')(aux_dense2)
        return Model( z_input, aux_out )
