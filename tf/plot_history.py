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
    with open(fname,'r') as f:
        data = pickle.load(f)
    return data

def plotHistory( loss, acc, val_loss, val_acc ):
    #['val_categorical_accuracy', 'loss', 'categorical_accuracy', 'val_loss']

    # summarize history for accuracy
    plt.figure(1,figsize=(10, 14), dpi=80)
    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('metrics.png')
    plt.show()

def plotHistoryWithError( loss, acc, val_loss, val_acc ):

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
    plt.subplot(2, 1, 1)
    plt.errorbar(range(len(acc_avg)),acc_avg, yerr=acc_err)
    plt.errorbar(range(len(val_acc_avg)),val_acc_avg, yerr=val_acc_err)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.errorbar(range(len(loss_avg)),loss_avg, yerr=loss_err)
    plt.errorbar(range(len(val_loss_avg)),val_loss_avg, yerr=val_loss_err)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('metrics.png')
    plt.show()

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def runTests(args):
    print( "Args: {}".format( args ) )
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.')
    parser.add_argument('history', nargs=1, help='A pickle file containing history from a drive training run')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    data = loadHistory( args.history[0] )

    if len(data) == 1:
        hist0 = data[1]
        plotHistory( hist0['loss'], hist0['categorical_accuracy'], hist0['val_loss'], hist0['val_categorical_accuracy'] )
    elif len(data) > 1:
        loss=[]
        acc=[]
        val_loss=[]
        val_acc=[]
        for hist in data:
            loss.append(hist['loss'])
            acc.append(hist['categorical_accuracy'])
            val_loss.append(hist['val_loss'])
            val_acc.append(hist['val_categorical_accuracy'])
        #for i in range(len(loss)):
        #    plt.plot( loss[i] )
        #plt.show()
        plotHistoryWithError( loss, acc, val_loss, val_acc )
