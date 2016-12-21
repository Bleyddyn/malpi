from time import time
import numpy as np
import matplotlib.pyplot as plt
from malpi.cnn import *
from malpi.data_utils import get_CIFAR10_data
from malpi.solver import Solver

def plot_solver(solver):
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

def getCIFAR10(verbose=True):
    data = get_CIFAR10_data()
    if verbose:
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    return data

reg = 3.37091767808e-05
lr = 0.000182436504066
decay = 1.0

#layers = ["conv-64", "Conv-64", "maxpool", "FC-1000", "fc-1000"]
#layer_params = [{'stride':1, 'dropout':0.5}, {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2}, (), ()]
layers = ["conv-32", "Conv-32", "maxpool", "FC-1000", "fc-10"]
layer_params = [{'stride':1, 'filter_size':7}, {'filter_size':7}, {'pool_stride':2, 'pool_width':2, 'pool_height':2}, (), ()]
model = MalpiConvNet(layers, layer_params, reg=reg)

data = getCIFAR10()
solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                lr_decay=decay,
                update_rule='adam',
                optim_config={
                  'learning_rate': lr,
                },
                verbose=True, print_every=50)
solver.train()
exit()

best_val_acc = 0.0
best_model = None
best_solver = None

#reguls = 10 ** np.random.uniform(-5, -4, 2) #[0.0001, 0.001, 0.01]
#lrs = 10 ** np.random.uniform(-6, -3, 5) #[1e-4, 1e-3, 1e-2]
#reguls = np.append([3.37091767808e-05],reguls)
#lrs = np.append([0.000182436504066],lrs)
variations = np.array([0.9,1.0,1.1])
reguls = np.array([3.37091767808e-05]) * variations
lrs = np.array([0.000182436504066]) * variations
features = [48] #[8, 16, 32, 48, 64]
decays = [1.0]
val_accs = []
hdims = 1000

"""
Try: Conv-64, Conv-64, maxpool, conv-128, conv-128, maxpool, conv-256, conv-256, maxpool, conv-512, conv-512, maxpool, conv-512, conv-512, maxpool, FC-4096, FC-4096, FC-1000, softmax
Dropout?
"""

for reg in reguls:
    for lr in lrs:
        for feat in features:
            for decay in decays:
                model = MultiLayerConvNet(num_filters=feat, filter_size=3, weight_scale=0.001, hidden_dim=hdims, reg=reg, dropout=0.5)


                solver = Solver(model, data,
                                num_epochs=6, batch_size=50,
                                lr_decay=decay,
                                update_rule='adam',
                                optim_config={
                                  'learning_rate': lr,
                                },
                                verbose=False, print_every=50)
                t_start = time()
                solver.train()
                val_accs.append(solver.best_val_acc)
                if solver.best_val_acc > best_val_acc:
                    best_val_acc = solver.best_val_acc
                    best_model = model
                    best_solver = solver
                print 'acc\t#filts\thdims\treg\tlr\tTime:'
                print '%f\t%d\t%d\t%f\t%f\t%fm' % (solver.best_val_acc,feat,hdims,reg,lr,(time() - t_start)/60.0)
                #plot_solver(solver)
plot_solver(best_solver)
#print('\a') # Sound a bell
#print('\a')
#print('\a')


print best_solver.best_val_acc
print best_model.reg
print best_solver.optim_config['learning_rate']
print val_accs
