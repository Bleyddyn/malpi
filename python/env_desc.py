import numpy as np
import gym

env_name = "MountainCar-v0"

env = gym.make(env_name)
print( "Action Space: %s" % (env.action_space,) )
if hasattr(env, 'get_action_meanings'):
    print( "   Meanings : %s" % (env.get_action_meanings(),) )
print( "Observation Space: %s" % (env.observation_space,) )
print( "   High: %s" % (env.observation_space.high,) )
print( "    Low: %s" % (env.observation_space.low,) )

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
print( "    Min: %s" % (np.min(observation_examples, axis=0),) )
print( "    Max: %s" % (np.max(observation_examples, axis=0),) )
print( "   Mean: %s" % (np.mean(observation_examples, axis=0),) )
print( "    Std: %s" % (np.std(observation_examples, axis=0),) )

state = env.reset()
print( "  Shape: %s" % (state.shape,) )

