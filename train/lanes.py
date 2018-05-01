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
    basename = "{}_{}x{}".format( prefix, size[0], size[1] )
    im_file = os.path.join( drive_dir, basename+".npy" )
    if os.path.exists(im_file):
        images = np.load(im_file)
    else:
        im_file = os.path.join( drive_dir, basename+".pickle" )
        with open(im_file,'r') as f:
            images = pickle.load(f)

    return images

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

def stats( onedir, auxName ):
    order = ["OutsideTrack", "OutsideLine", "LeftLane", "CenterLine", "RightLane", "InsideLine", "InsideTrack"]
    auxFile = os.path.join( onedir, "{}_aux.npy".format(auxName ) )
    if os.path.exists(auxFile):
        actions = np.load(auxFile)
        actions = actions.astype('str')
    stats = defaultdict(int)
    for label in actions:
        stats[str(label)] += 1
    statstr = ""
    for lane in order:
        statstr += str(stats[lane]) + "\t"
    print( "{}\t{}\t{}".format( onedir, len(actions), statstr ) )

def rewards( onedir ):
    order = [-10.0, -5.0, 0.5, 0.5, 1.0, -5.0, -10.0]
    lanesFile = os.path.join( onedir, "lanes_pred.npy" )
    lanes = np.load(lanesFile)
    lanes = np.argmax(lanes, axis=1)
    reward = 0.0
    for lane in lanes:
        reward += order[lane]
    print( "{}\t{}\t{}".format( onedir, len(lanes), reward ) )

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
    parser.add_argument('--stats', action="store_true", default=False, help='Read the aux files and count total number of labels and count each label')
    parser.add_argument('--aux', default="LanePosition", help='Use this auxiliary data in place of standard actions')
    parser.add_argument('--rewards', action="store_true", default=False, help='Calculate rewards based on predicted lane position')
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

    if args.stats and args.aux is None:
        parser.print_help()
        print( "\n--stats requires --aux" )
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
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[metrics.categorical_accuracy] )

    if args.stats:
        print( "Drive	Count	OutsideTrack	OutsideLine	LeftLane	CenterLine	RightLane	InsideLine	InsideTrack" )

    # Redo this so that we load, evaluate and output results for one directory at a time
    for onedir in args.dirs:
        if args.stats:
            stats( onedir, args.aux )
        elif args.rewards:
            rewards( onedir )
        else:
            print( "{}".format( onedir ) )
            prefix = "images"
            if args.pred is not None:
                prefix = args.pred
            images = loadOneDrive( onedir, prefix=prefix )
            images = images.astype(np.float)
            normalize( images )

            if args.pred is not None:
                out = model.predict( x=images )
                basename = "{}_pred.npy".format( args.pred )
                pred_file = os.path.join( onedir, basename )
                np.save( pred_file, out )
            else:
                auxData = loadAuxData( [onedir], args.aux )
                out = model.evaluate( x=images, y=auxData )
                print( "{} {}: {}".format( onedir, model.metrics_names, out ) )

    #msg2 = "Model " + args.name
    #notify.notify( "Lane testing complete", subTitle=msg2, message=msg, email_to=args.notify )
