import pickle
import argparse
import numpy as np

from hyperopt import hp, STATUS_OK, Trials
from hyperopt.pyll_utils import expr_to_config
import hyperopt

import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def plotRegressionHparam( losses, values, name, logx=False ):

    plt.figure(1,figsize=(10, 14), dpi=80)
    #plt.suptitle( name, fontsize=16 )
    if logx:
        plt.semilogx(values, losses, '.')
    else:
        plt.plot(values, losses, '.')
    plt.title(name)
    plt.ylabel('val acc')
    plt.xlabel('value')
    plt.show()

def plotCategoricalHparam( category_dict, name ):
    labels = category_dict.keys()
    N = len(labels)
    means = []
    std = []
    for key in labels:
        values = category_dict[key]
        means.append( np.mean(values) )
        std.append( np.std(values) )

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, means, width, color='r', yerr=std)

# add some text for labels, title and axes ticks
    ax.set_ylabel('val acc')
    ax.set_title(name)
    ax.set_xticks(ind + (width/2.0))
    ax.set_xticklabels(labels)
    plt.show()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def plotTrials(filename):
    max_batch = 128
# This should be included in the trials object
    space = { 'learning_rate': hp.loguniform('learning_rate', -9, -4 ),
              'l2_reg': hp.loguniform('l2_reg', -10, -3 ),
              'batch_size': hp.quniform('batch_size', 5, max_batch, 1),
              'dropouts': hp.choice('dropouts', ["low","mid","high","up","down"]),
              'optimizer': hp.choice('optimizer', ["RMSProp", "Adagrad", "Adadelta", "Adam"]),
              'epochs': 40 }

#conditions = ()
#hps = {}
#expr_to_config(space, conditions, hps)

#print( "{}".format( hps ) )
#dropouts = space['dropouts']
#print( "{}".format( dropouts ) )

    with open(filename,'r') as f:
        trials = pickle.load(f)

#print( "trials.argmin: {}".format( trials.argmin ) )
#print( "eval: {}".format( hyperopt.space_eval( space, trials.argmin ) ) )

    losses = []
    hparams = {}

    for trial in trials.trials:
# >>> trial.keys()
# ['refresh_time', 'book_time', 'misc', 'exp_key', 'owner', 'state', 'version', 'result', 'tid', 'spec']
# trial['misc'] = {'tid': 0, 'idxs': {'optimizer': [0], 'learning_rate': [0], 'batch_size': [0], 'timesteps': [0], 'dropouts': [0], 'l2_reg': [0]}, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'vals': {'optimizer': [0], 'learning_rate': [0.00021416260507967771], 'batch_size': [6.0], 'timesteps': [18.0], 'dropouts': [0], 'l2_reg': [0.005126253249084938]}, 'workdir': None}
# trial['exp_key']
# trial['owner']
# trial['state'] = 2
# trial['version'] = 0
# trial['tid'] = 0
# trial['spec']
# trial['result'].keys() = ['status', 'loss', 'val', 'history']
        loss = trial['result']['val']
        losses.append(loss)

        argvals = trial['misc']['vals']
        for key, val in argvals.iteritems():
            argvals[key] = val[0]
        hparam1 = hyperopt.space_eval( space, argvals )
#    print( 'trial1: {}'.format( hparam1 ) )
# trial1: {'epochs': 40, 'optimizer': 'RMSProp', 'learning_rate': 0.00021416260507967771, 'dropouts': 'low', 'batch_size': 6.0, 'l2_reg': 0.005126253249084938}
        for key, val in hparam1.iteritems():
            if isinstance(val, basestring):
                if key not in hparams:
                    hparams[key] = { val: [] }
                if val not in hparams[key]:
                    hparams[key][val] = []
                hparams[key][val].append(loss)
            else:
                if key not in hparams:
                    hparams[key] = []
                hparams[key].append(val)


#print( "final: {}".format( hparams ) )
    plotCategoricalHparam( hparams['dropouts'], 'dropouts' )
    plotCategoricalHparam( hparams['optimizer'], 'optimizer' )
    plotRegressionHparam( losses, hparams['batch_size'], 'Batch Size' )
    plotRegressionHparam( losses, hparams['learning_rate'], 'Learning Rate', logx=True )
    plotRegressionHparam( losses, hparams['l2_reg'], 'L2 Regularization', logx=True )

def plotCurrent(filename):
    acc = []
# Only read in the lines back to the marker at the beginning of the current run
    for line in reversed(open(filename).readlines()):
        if line.startswith("#"):
            break
        acc.insert(0,float(line))
    print( "Sorted: \n{}".format( sorted(acc) ) )

#acc = np.loadtxt('hparam_current.txt')
    x = range(len(acc))
    z = np.polyfit( x, acc, 2, rcond=None, full=False, w=None, cov=False)
    p2 = np.poly1d(np.polyfit(x, acc, 2))
    xp = np.linspace(0, len(acc), 100)
    plt.plot(x, acc, '-', xp, p2(xp), '--')
    plt.savefig( 'hparam_current.png')
    plt.show()

def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Plot results from a hyperparameter optimization run.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', help='Plot results from a pickled hyperopt Trials object.')
    parser.add_argument('--current', default="hparam_current.txt", help='Plot results from the current run.')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    if args.trial is not None:
        plotTrials(args.trial)
    else:
        plotCurrent(args.current)
