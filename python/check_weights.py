import numpy as np
import pickle
from malpi.optimizer import *
from malpi import optim

def stats(arr, msg=""):
    mi = np.min(arr)
    ma = np.max(arr)
    av = np.mean(arr)
    std = np.std(arr)
    arr_abs = np.abs(arr)
    mi_abs = np.min(arr_abs)
    ma_abs = np.max(arr_abs)
    print "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs)

with open('dqn_pong_v5.pickle') as f:
    model = pickle.load( f )

for i in range(10):
    state = np.random.uniform( 0.0, 1.0, (1,4,84,84) )
    q_values, _ = model.forward(state, mode="test")
    print q_values[0]

#with open('optimizer_test.pickle') as f:
#    (w,dw,config) = pickle.load( f )
#
#del config['cache']
#
#update_rule = optim.rmsprop
#
#model.params = {'W5': w}
#optim = Optimizer("rmsprop", model, learning_rate=config['learning_rate'], decay_rate=config['decay_rate'], epsilon=config['epsilon'])
#print config
#optim.describe()
#
#diff = model.params['W5'] - w
#stats(diff, 'before ')
#
#next_w, next_config = update_rule(w, dw, config)
#
#grads = {'W5': dw}
#optim.update(grads)
#
#diff = model.params['W5'] - next_w
#stats(diff, 'after  ')
#diff = optim.cache['W5'] - next_config['cache']
#stats(optim.cache['W5'], 'cache  ')
#stats(diff, 'diffs  ')
#
#if False:
#    for k,w in model.params.iteritems():
#        print k
#        mask_zeros = w != 0.0
#        mask = np.abs(w) < 1e-20
#        mask = np.logical_and(mask_zeros,mask)
#        if np.count_nonzero(mask) > 0:
#            print "Underflow in %s " % (k,)
