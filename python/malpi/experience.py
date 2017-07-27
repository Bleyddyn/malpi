import numpy as np
from collections import deque, namedtuple
import random

class Experience(object):
    Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state", "action_probs"])
    
    def __init__( self, maxN, state_dim ):
        self.N = maxN
        sdim = (maxN,) + state_dim
        self.states = np.zeros(sdim)
        self.actions = np.zeros(maxN)
        self.rewards = np.zeros(maxN)
        self.done = np.ones(maxN).astype(np.float)
        self.next_states = np.zeros(sdim)
        self.next_insert = 0
        self.max_batch = 0

    def size( self ):
        return self.max_batch

    def save( self, state, action, reward, done, next_state ):
        self.states[self.next_insert,:] = state
        self.actions[self.next_insert] = action
        self.rewards[self.next_insert] = reward
        if done:
            self.done[self.next_insert] = 0.0
        self.next_states[self.next_insert,:] = next_state
        self.next_insert += 1
        self.max_batch = max(self.next_insert, self.max_batch)
        if self.next_insert >= self.N:
            self.next_insert = 0

    def batch( self, batch_size ):
        if batch_size >= self.max_batch:
            return (None,None,None,None)
        start = np.random.randint( 0, high=(1+self.max_batch - batch_size) )
        end = start + batch_size
        r_s = self.states[start:end,:]
        r_a = self.actions[start:end]
        r_r = self.rewards[start:end]
        r_d = self.done[start:end]
        r_n = self.next_states[start:end,:]
        return (r_s,r_a,r_r,r_d,r_n)

    def clear( self ):
        self.next_insert = 0
        self.max_batch = 0

    @staticmethod
    def test():
        e = Experience( 10, (20,20) )
        s = np.zeros( (20,20) )
        e.save( s, 1, -3, False, s )
        e.save( s, 2, -4, False, s )
        e.save( s, 3, -5, False, s )
        e.save( s, 4, -6, False, s )
        print e.max_batch # 4
        s1, a, r, d, n = e.batch( 2 )
        print s1.shape # (2, 20, 20)
        print a # e.g. [ 1.  2.]
        print r # e.g. [-3. -4.]
        for _ in range(2):
            e.save( s, 5, -7, False, s )
            e.save( s, 6, -8, False, s )
            e.save( s, 7, -9, True, s )
            e.save( s, 8, -10, False, s )
        print e.max_batch # 10
        print e.actions[0:2]
        print e.rewards[0:2]

class Experience2(object):

    Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state", "action_probs"])

    def __init__( self, maxN, stateDim=None ):
        # stateDim is only for compatibility with Experience
        self.maxN = maxN
        self.memory = deque(maxlen=maxN)
        self.priority = []

    def __len__(self):
        return len(self.memory)

    def size( self ):
        return len(self.memory)

    def save( self, state, action, reward, done, next_state, action_probs=None ):
        if done:
            done = 0.0
        else:
            done = 1.0

        self.memory.append(self.Transition(state, action, reward, done, next_state, action_probs))

    def batch( self, batch_size ):
        if batch_size > len(self.memory):
            return (None,None,None,None,None,None)

        samples = random.sample(self.memory, batch_size)
        states_batch, action_batch, reward_batch, done_batch, next_states_batch, probs_batch = map(np.array, zip(*samples))
        return (states_batch, action_batch, reward_batch, done_batch, next_states_batch, probs_batch)

    def all( self, max_size=200 ):
        if len(self.memory) > max_size:
            states_batch, action_batch, reward_batch, done_batch, next_states_batch, probs_batch = map(np.array, zip(*self.memory[-200:]))
        else:
            states_batch, action_batch, reward_batch, done_batch, next_states_batch, probs_batch = map(np.array, zip(*self.memory))
        return (states_batch, action_batch, reward_batch, done_batch, next_states_batch, probs_batch)

    def clear( self ):
        self.memory.clear()

    @staticmethod
    def test():
        e = Experience2( 10 )
        s = np.zeros( (20,20) )
        e.save( s, 1, -3, False, s )
        e.save( s, 2, -4, True, s )
        e.save( s, 3, -5, False, s )
        e.save( s, 4, -6, True, s )
        s1, a, r, d, n, probs = e.batch( 2 )
        print s1.shape # (2, 20, 20)
        print a # e.g. [ 1.  2.]
        print r # e.g. [-3. -4.]
        print d
        for _ in range(2):
            e.save( s, 5, -7, False, s )
            e.save( s, 6, -8, False, s )
            e.save( s, 7, -9, True, s )
            e.save( s, 8, -10, False, s )

