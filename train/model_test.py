"""
"""
from __future__ import print_function
import os
import sys
import pickle
import json
from time import time
import argparse
from collections import defaultdict

import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model_keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import keras.backend as K
from keras.models import model_from_json
from keras import metrics

import experiment
import notify
from load_aux import loadAuxData

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

def setCPUCores( cores ):
    # Actual device_count seems to have less effect than number of threads
    config = tf.ConfigProto(intra_op_parallelism_threads=cores, inter_op_parallelism_threads=cores,
                            allow_soft_placement=True, device_count = {'CPU': cores})
    set_session(tf.Session(config=config))

def loadOneDrive( drive_dir, size=(120,120), prefix="images" ):
    actions_file = os.path.join( drive_dir, "image_actions.npy" )
    if os.path.exists(actions_file):
        actions = np.load(actions_file)

    basename = "{}_{}x{}".format( prefix, size[0], size[1] )
    im_file = os.path.join( drive_dir, basename+".npy" )
    if os.path.exists(im_file):
        images = np.load(im_file)
    else:
        im_file = os.path.join( drive_dir, basename+".pickle" )
        with open(im_file,'r') as f:
            images = pickle.load(f)

    return images, actions

def normalize( images ):
    rmean = 92.93206363205326
    gmean = 85.80540021330793
    bmean = 54.14884297660608
    rstd = 57.696159704394354
    gstd = 53.739380109203445
    bstd = 47.66536771313241

    images[:,:,:,0] -= rmean
    images[:,:,:,1] -= gmean
    images[:,:,:,2] -= bmean
    images[:,:,:,0] /= rstd
    images[:,:,:,1] /= gstd
    images[:,:,:,2] /= bstd

    return images

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Run a trained lane detector on images and compare with any existing labels.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Path to the model json file')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')
    parser.add_argument('--pred', help='For each directory, run the model on an image file with this prefix and write predictions to a new file in the same directory')
    parser.add_argument('--categorical', action="store_true", default=False, help='Model has a categorical action space')
    parser.add_argument('--notify', help='Email address to notify when the training is finished')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            tmp_dirs = f.read().split('\n')
            args.dirs.extend(tmp_dirs)

    if len(args.dirs) == 0 and not args.test_only:
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    for i in reversed(range(len(args.dirs))):
        if args.dirs[i].startswith("#"):
            del args.dirs[i]
        elif len(args.dirs[i]) == 0:
            del args.dirs[i]
            
    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    K.set_learning_phase(True)
    setCPUCores( 4 )

    with open(args.model, "r") as jfile:
        model = model_from_json(jfile.read())

    weights = os.path.splitext(args.model)[0] + '_weights.h5'
    model.load_weights(weights)
    if args.categorical:
        loss='categorical_crossentropy'
        metrics_arr=[metrics.categorical_accuracy]
    else:
        loss='mse'
        metrics_arr=['mse']
    model.compile(loss=loss, optimizer="adam", metrics=metrics_arr )

    for onedir in args.dirs:
        print( "{}".format( onedir ) )
        images, actions = loadOneDrive( onedir )
        images = images.astype(np.float)
        normalize( images )

        out = model.predict( x=images )
        #basename = "{}_pred.npy".format( args.pred )
        #pred_file = os.path.join( onedir, basename )
        #np.save( pred_file, out )
# For each action plot out - actions
