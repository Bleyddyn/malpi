import os
import pickle
from time import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

def describeDriveData( data ):
    print( data.keys() )
    for key, value in data.iteritems():
        try:
            print( "{} length {}".format( key, len(value) ) )
        except:
            pass

def loadHistory( fname="histories.pickle" ):
    with open(fname,'rb') as f:
        data = pickle.load(f)
    return data

def plotHistory( loss, acc, val_loss, val_acc, name, plot_dir ):
    #['val_categorical_accuracy', 'loss', 'categorical_accuracy', 'val_loss']

    # summarize history for accuracy
    plt.figure(1,figsize=(10, 14), dpi=80)
    plt.suptitle( name, fontsize=16 )
    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.ylim(0.8,4.0)

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    #plt.semilogy( loss )
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.ylim(0.05,0.2)
    #plt.ylim(0.04,1.0)
    plt.savefig( os.path.join( plot_dir, name.replace(' ', '_') + '.png' ) )
    plt.show()

def plotHistoryWithError( loss, acc, val_loss, val_acc, name, plot_dir ):

    loss_avg = np.mean(loss, axis=0)
    loss_err = np.std(loss, axis=0)
    acc_avg = np.mean(acc, axis=0)
    acc_err = np.std(acc, axis=0)
    val_loss_avg = np.mean(val_loss, axis=0)
    val_loss_err = np.std(loss, axis=0)
    val_acc_avg = np.mean(val_acc, axis=0)
    val_acc_err = np.std(val_acc, axis=0)

    # summarize history for accuracy
    plt.figure(1,figsize=(10, 14), dpi=80)
    plt.suptitle( name, fontsize=16 )
    plt.subplot(2, 1, 1)
    plt.errorbar(range(len(acc_avg)),acc_avg, yerr=acc_err)
    plt.errorbar(range(len(val_acc_avg)),val_acc_avg, yerr=val_acc_err)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.errorbar(range(len(loss_avg)),loss_avg, yerr=loss_err)
    plt.errorbar(range(len(val_loss_avg)),val_loss_avg, yerr=val_loss_err)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig( os.path.join( plot_dir, name.replace(' ', '_') + '.png' ) )
    plt.show()

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def plotDagger():
    files = ["experiments/Track_CW_FC_DK_256_opt2/histories.pickle", "experiments/Track_CW_FC_DK_256_dag1/histories.pickle", "experiments/Track_CW_FC_DK_256_dag2/histories.pickle", "experiments/Track_CW_FC_DK_256_dag3/histories.pickle", "experiments/Track_CW_FC_DK_256_dag4/histories.pickle"]

    plt.figure(1,figsize=(10, 14), dpi=80)
    train = plt.subplot(2, 1, 1)
    val = plt.subplot(2, 1, 2)
    lines = []

    for fname in files:
        data = loadHistory( fname )
        acc=[]
        val_acc=[]
        for hist in data:
            acc.append(hist['categorical_accuracy'])
            val_acc.append(hist['val_categorical_accuracy'])
        acc_avg = np.mean(acc, axis=0)
        acc_err = np.std(acc, axis=0)
        val_acc_avg = np.mean(val_acc, axis=0)
        val_acc_err = np.std(val_acc, axis=0)

        line = train.errorbar(range(len(acc_avg)),acc_avg, yerr=acc_err)
        val.errorbar(range(len(val_acc_avg)),val_acc_avg, yerr=val_acc_err)
        lines.append(line)

    plt.figlegend(lines, ['Opt2', 'Dag1', 'Dag2', 'Dag3', 'Dag4'], loc = 'center', ncol=5 )
    plt.subplots_adjust(hspace=0.4)
    train.set_title('Training Accuracy')
    train.set_ylabel('accuracy')
    train.set_xlabel('epoch')
    val.set_title('Validation Accuracy')
    val.set_ylabel('accuracy')
    val.set_xlabel('epoch')
    plt.savefig( 'dagger_plot.png' )
    plt.show()

def runTests(args):
    print( "Args: {}".format( args ) )
    pass

def getExpName( exp_dir ):
    if exp_dir[-1] == "/":
        exp_dir = exp_dir[:-1]
    name = os.path.basename(exp_dir)
    meta = os.path.join(exp_dir,name+".txt")
    if os.path.exists(meta):
        with open(meta,'r') as f:
            for line in f:
                if line.startswith('Name: '):
                    name = line[6:]
                    break
    return name.strip()

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.')
    parser.add_argument('history', nargs='?', help='A pickle file containing history from a drive training run')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
    parser.add_argument('--name', help='name to use for plot title and filename')
    parser.add_argument('--exp', help='Directory with saved experiment meta-data, including a history file')
    parser.add_argument('--dagger', action="store_true", default=False, help='Plots for first Dagger blog post')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    if args.dagger:
        plotDagger()
        exit()

    fname = 'histories.pickle'
    plot_name = 'metrics'
    plot_dir = '.'
    if args.exp is not None:
        fname = os.path.join( args.exp, fname)
        plot_name = getExpName(args.exp)
        plot_dir = args.exp
    else:
        if args.history is not None:
            fname = args.history

    if args.name is not None:
        plot_name = args.name

    data = loadHistory( fname )

    if len(data) == 1:
        hist0 = data[0]
        if 'categorical_accuracy' in hist0:
            plotHistory( hist0['loss'], hist0['categorical_accuracy'], hist0['val_loss'], hist0['val_categorical_accuracy'], plot_name, plot_dir )
        elif 'mean_squared_error' in hist0:
#        dict_keys(['val_loss', 'val_mean_squared_error', 'loss', 'mean_squared_error'])
            plotHistory( hist0['loss'], hist0['mean_squared_error'], hist0['val_loss'], hist0['val_mean_squared_error'], plot_name, plot_dir )
    elif len(data) > 1:
        loss=[]
        acc=[]
        val_loss=[]
        val_acc=[]
        for hist in data:
            print( hist.keys() )
            loss.append(np.array(hist['loss']))
            val_loss.append(np.array(hist['val_loss']))
            if 'categorical_accuracy' in hist:
                acc.append(np.array(hist['categorical_accuracy']))
                val_acc.append(np.array(hist['val_categorical_accuracy']))
            elif 'mean_squared_error' in hist:
                acc.append(np.array(hist['mean_squared_error']))
                val_acc.append(np.array(hist['val_mean_squared_error']))

        loss = np.array(loss)
        acc = np.array(acc)
        val_loss = np.array(val_loss)
        val_acc = np.array(val_acc)

        #for i in range(len(loss)):
        #    plt.plot( loss[i] )
        #plt.show()
        plotHistoryWithError( loss, acc, val_loss, val_acc, plot_name, plot_dir )
