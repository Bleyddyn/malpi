from time import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from malpi.cnn import *
from malpi.data_utils import get_CIFAR10_data
from malpi.solver import Solver
from optparse import OptionParser

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
    data = get_CIFAR10_data(num_training=49000)
    if verbose:
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    return data

def log( message, name='test' ):
    logFileName = name + ".log"
    fmt = '%Y-%m-%d-%H-%M-%S'
    datestr = datetime.datetime.now().strftime(fmt)

    with open(logFileName,'a') as outf:
        outf.write(datestr + ": " + message + "\n")

def hyperparameterGenerator( oneRun = False ):
    variations = np.array([0.9,1.0,1.1])
    if oneRun:
        reguls = [3.37091767808e-05]
        lrs = [0.0002006801544726]
    else:
        reguls = np.array([3.37091767808e-05]) * variations
        lrs = np.array([0.000182436504066]) * variations
#reguls = 10 ** np.random.uniform(-5, -4, 2) #[0.0001, 0.001, 0.01]
#lrs = 10 ** np.random.uniform(-6, -3, 5) #[1e-4, 1e-3, 1e-2]
#reguls = np.append([3.37091767808e-05],reguls)
#lrs = np.append([0.000182436504066],lrs)

    decays = [1.0]

    for reg in reguls:
        for lr in lrs:
            for decay in decays:
                hparams = { "reg": reg, "lr": lr, "lr_decay":decay, "epochs":6, "batch_size":50, "update":"adam" }
                yield hparams

def train():
    name = "ThreeLayerTest1"
    layers = ["conv-8", "maxpool", "conv-16", "maxpool", "conv-32", "fc-10"]
    layer_params = [{'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3},
        {'relu':False}]

    log( "%s = %s" % (name, str(layers)), name )
    log( "   %s" % (str(layer_params,)), name )
    data = getCIFAR10(verbose=False)

    model_name = name + ".pickle"

    val_accs = []
    best_solver = None
    best_val_acc = 0.0
    best_model = load_malpi( model_name, verbose=False)
    if best_model:
        best_val_acc = best_model.validation_accuracy

    for hparams in hyperparameterGenerator(oneRun=True):
        model = MalpiConvNet(layers, layer_params, reg=hparams['reg'], dtype=np.float16, verbose=False)
        model.hyper_parameters = hparams
        solver = Solver(model, data,
                        num_epochs=hparams['epochs'], batch_size=hparams['batch_size'],
                        lr_decay=hparams['lr_decay'],
                        update_rule=hparams['update'],
                        optim_config={
                          'learning_rate': hparams['lr'],
                        },
                        verbose=True, print_every=50)

        log( "Started training model: %s" % (name,), name=name )
        log( "   Hyper-parameters: %s" % (str(hparams),), name=name )
        solver.train()
        log( "   Validation Accuracy: %f" % (solver.best_val_acc,) , name=name )
        log( "Finished training", name=name )

        val_accs.append(solver.best_val_acc)
        if solver.best_val_acc > best_val_acc:
            best_val_acc = solver.best_val_acc
            best_model = model
            best_solver = solver

    log( "", name=name )

    best_model.name = name
    best_model.validation_accuracy = best_val_acc
    best_model.save(model_name)

    #plot_solver(best_solver)
    print val_accs
#    print('\a') # Sound a bell
#    print('\a')
#    print('\a')

def classify(data):
    model = load_malpi('SimpleTest1.pickle')
    scores = model.loss(data)
    print scores

def testload():
    model = load_malpi('SimpleTest1.pickle')
    data = getCIFAR10(verbose=False)
    solver = Solver(model, data)
    train_acc = solver.check_accuracy(data["X_train"], data["y_train"], num_samples=1000)
    val_acc = solver.check_accuracy(data["X_val"], data["y_val"])
    print "train acc: %f; val_acc: %f" % (train_acc,val_acc)

#Try: Conv-64, Conv-64, maxpool, conv-128, conv-128, maxpool, conv-256, conv-256, maxpool, conv-512, conv-512, maxpool, conv-512, conv-512, maxpool, FC-4096, FC-4096, FC-1000, softmax

def describeModel( name ):
    model = load_malpi(name+'.pickle')
#    if not hasattr(model, 'hyper_parameters'):
#        model.hyper_parameters = {}
    model.describe()
#    model.save(name+'.pickle')

def getOptions():
    parser = OptionParser()
    parser.add_option("-d","--describe",dest="name",help="Describe a model saved in a pickle file: <name>.pickle");
    (options, args) = parser.parse_args()
    return (options, args)

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.name:
        describeModel(options.name)
    else:
        train()
