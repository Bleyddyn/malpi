from gym.envs.registration import register

from malpi.dkwm.gym_envs.dkwm_env import DKWMEnv

register(
    id='dkwm-v0',
    entry_point='malpi.dkwm.gym_envs:DKWMEnv',
    )
