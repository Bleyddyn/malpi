import numpy as np
import pickle
from malpi.optimizer import *
from malpi import optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stats(arr, msg=""):
    mi = np.min(arr)
    ma = np.max(arr)
    av = np.mean(arr)
    std = np.std(arr)
    arr_abs = np.abs(arr)
    mi_abs = np.min(arr_abs)
    ma_abs = np.max(arr_abs)
    print "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs)

with open('dqn_mc_v2.pickle') as f:
    model = pickle.load( f )

xs = []
ys = []
zs = []
zs2 = []
zs3 = []

for x in np.random.uniform( -1.3, 0.7, 20 ):
    for y in np.random.uniform( -0.8, 0.8, 20 ):
        qvalues,_ = model.forward( np.reshape( np.array([x, y]), (1,2)), mode="test")
        xs.append(x)
        ys.append(y)
        zs.append( qvalues[0][0] )
        zs2.append( qvalues[0][1] )
        zs3.append( qvalues[0][2] )

fig = plt.figure()
ax = fig.add_subplot(311, projection='3d')
ax.scatter(xs,ys,zs=zs)
ax.set_xlabel('Location')
ax.set_ylabel('Velocity')
ax.set_title('Action Left')

ax = fig.add_subplot(312, projection='3d')
ax.scatter(xs,ys,zs=zs2)
ax.set_xlabel('Location')
ax.set_ylabel('Velocity')
ax.set_title('Action Noop')

ax = fig.add_subplot(313, projection='3d')
ax.scatter(xs,ys,zs=zs3)
ax.set_xlabel('Location')
ax.set_ylabel('Velocity')
ax.set_title('Action Right')

plt.show()

# get qvalues for a range of mc inputs and plot them
#High: [ 0.6   0.07]
#Low: [-1.2  -0.07]

#for i in range(10):
#    state = np.random.uniform( 0.0, 1.0, (1,4,84,84) )
#    q_values, _ = model.forward(state, mode="test")
#    print q_values[0]

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
