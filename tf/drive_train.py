import os
import pickle
from time import time
import argparse

import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model_keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras.backend as K

def describeDriveData( data ):
    print( data.keys() )
    for key, value in data.iteritems():
        try:
            print( "{} length {}".format( key, len(value) ) )
        except:
            pass

def embedActions( actions ):
    embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
    emb = []
    prev_act = 0
    for act in actions:
        if not act.startswith("speed"):
            prev_act = embedding[act]
            emb.append( embedding[act] )
        else:
            emb.append( prev_act )
    return emb

def setCPUCores( cores ):
    # Actual device_count seems to have less effect than number of threads
    config = tf.ConfigProto(intra_op_parallelism_threads=cores, inter_op_parallelism_threads=cores,
                            allow_soft_placement=True, device_count = {'CPU': cores})
    set_session(tf.Session(config=config))

def loadOneDrive( drive_dir ):
    drive_file = os.path.join( drive_dir, "drive.pickle" )

    with open(drive_file,'r') as f:
        data = pickle.load(f)
        #data = pickle.load(f fix_imports=True, encoding='bytes')
    actions = data['image_actions']

    im_file = os.path.join( drive_dir, "images_120x120.pickle" )
    with open(im_file,'r') as f:
        images = pickle.load(f)

    return images, actions

def printLearningRate(model):
    optimizer = model.optimizer
    lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * tf.cast(optimizer.iterations, tf.float32) )))
    print('      LR: {:.6f}'.format(lr))
    
class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

def step_decay(epoch):
# Usage: lrate = LearningRateScheduler(step_decay)
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  math.floor((1+epoch)/epochs_drop))
    return lrate

def exp_decay(epoch):
# Usage: lrate = LearningRateScheduler(exp_decay)
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * exp(-k*t)
    return lrate

def loadData( dirs, image_norm=True ):
    images = []
    actions = []

    for onedir in dirs:
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir )
            images.extend(dimages)
            actions.extend(dactions)

    images = np.array(images)
    images = images.astype(np.float) # / 255.0

    if image_norm:
# should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
        images[:,:,:,0] -= np.mean(images[:,:,:,0])
        images[:,:,:,1] -= np.mean(images[:,:,:,1])
        images[:,:,:,2] -= np.mean(images[:,:,:,2])
        images[:,:,:,0] /= np.std(images[:,:,:,0])
        images[:,:,:,1] /= np.std(images[:,:,:,1])
        images[:,:,:,2] /= np.std(images[:,:,:,2])

    y = embedActions( actions )
    y = to_categorical( y, num_classes=5 )
    return images, y

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def hparamsToArray( hparams ):
    out = []
    out.append( int(hparams.get( "timesteps", 10 )) )
    out.append( hparams.get( "l2_reg", 0.005 ) )

    dropouts = hparams.get( "dropouts", [0.2,0.2,0.2,0.2,0.2] )
    if "dropouts" in hparams:
        dropouts = hparams["dropouts"]
        if isinstance(dropouts, basestring):
            if dropouts == "low":
                dropouts = [0.2,0.2,0.2,0.2,0.2]
            elif dropouts == "mid":
                dropouts = [0.4,0.4,0.4,0.4,0.4]
            elif dropouts == "high":
                dropouts = [0.6,0.6,0.6,0.6,0.6]
            elif dropouts == "up":
                dropouts = [0.2,0.3,0.4,0.5,0.6]
            elif dropouts == "down":
                dropouts = [0.6,0.5,0.4,0.3,0.2]
    out.append( dropouts )

    out.append( hparams.get( "learning_rate", 0.003 ) )
    out.append( hparams.get( "validation_split", 0.20 ) )
    out.append( int(hparams.get( "batch_size", 5 )) )
    out.append( hparams.get( "optimizer", "RMSprop" ) )
    out.append( int(hparams.get( "epochs", 40 )) )

    return out

def hparamsToDict( hparams ):
    out = {}
    out["timesteps"] = hparams[0]
    out["l2_reg"] = hparams[1]
    out["dropouts"] = hparams[2]
    out["learning_rate"] = hparams[3]
    out["validation_split"] = hparams[4]
    out["batch_size"] = hparams[5]
    out["optimizer"] = hparams[6]
    out["epochs"] = hparams[7]
    return out

def makeOptimizer( optimizer, learning_rate ):
    # See: https://medium.com/towards-data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    if optimizer == "RMSProp":
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.005) # default lr=0.001
    elif optimizer == "Adagrad":
        optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0) # default lr=0.01
    elif optimizer == "Adadelta":
        optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0) # default lr=1.0
    elif optimizer == "Adam":
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # default lr=0.001

    return optimizer

def fitFC( input_dim, images, y, verbose=1, epochs=40, timesteps=10, l2_reg=0.005, dropouts=[0.25,0.25,0.25,0.25,0.25],
           learning_rate=0.003, validation_split=0.15, batch_size=32, optimizer="RMSprop" ):
    num_actions = len(y[0])
    callbacks = None

    if verbose:
        save_chk = ModelCheckpoint("weights_{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks = [save_chk]

    optimizer = makeOptimizer( optimizer, learning_rate )
    
    model = model_keras.make_model_test( num_actions, input_dim, optimizer=optimizer, dropouts=dropouts )

    history = model.fit( images, y, validation_split=validation_split, epochs=epochs, verbose=verbose, batch_size=batch_size, callbacks=callbacks )

    if verbose:
        model.save( 'best_model.h5' )
        model.save_weights('best_model_weights.h5')
        if X_val is not None and y_val is not None:
            (val_loss, val_acc) = evaluate( num_actions, input_dim, X_val, y_val, dropouts=dropouts )
            print( "Final Validation loss/acc: {}  {}".format( val_loss, val_acc) )

    running = runningMean(history.history['val_categorical_accuracy'], 5)
    max_running = np.max( running )
    print( "Max validation (rmean=5, at {}): {}".format( np.argmax(running), max_running ) )

    return (max_running, history)

def fitLSTM( input_dim, images, y, verbose=1, epochs=40, timesteps=10, l2_reg=0.005, dropouts=[0.25,0.25,0.25,0.25,0.25],
             learning_rate=0.003, validation_split=0.15, batch_size=5, optimizer="RMSprop" ):
    num_actions = len(y[0])
    num_samples = len(images)
    hold_out = (num_samples % timesteps) + (5 * timesteps)
    num_samples = num_samples - hold_out
    X_val = images[num_samples:,:]
    y_val = y[num_samples:,:]
    images = images[0:num_samples,:]
    images = np.reshape( images, (num_samples/timesteps, timesteps) + input_dim )
    y = y[0:num_samples,:]
    y = np.reshape( y, (num_samples/timesteps, timesteps, num_actions) )

    callbacks = None
#    if verbose:
#        save_chk = ModelCheckpoint("weights_{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#        callbacks = [save_chk]

    optimizer = makeOptimizer( optimizer, learning_rate )

    model = model_keras.make_model_lstm_fit( num_actions, input_dim, timesteps=timesteps, stateful=False, dropouts=dropouts, optimizer=optimizer )

    history = model.fit( images, y, validation_split=validation_split, epochs=epochs, verbose=verbose, batch_size=batch_size, shuffle=False, callbacks=callbacks )

    if verbose:
        model.save( 'best_model.h5' )
        model.save_weights('best_model_weights.h5')
#        if X_val is not None and y_val is not None:
#            (val_loss, val_acc) = evaluate( num_actions, input_dim, X_val, y_val, dropouts=dropouts )
#            print( "Final Validation loss/acc: {}  {}".format( val_loss, val_acc) )

    running = runningMean(history.history['val_categorical_accuracy'], 5)
    max_running = np.max( running )
    print( "Max validation (rmean=5, at {}): {}".format( np.argmax(running), max_running ) )

    return (max_running, history)

def evaluate( num_actions, input_dim, X_val, y_val, dropouts=[0.25,0.25,0.25,0.25,0.25] ):
    model2 = model_keras.make_model_lstm( num_actions, input_dim, batch_size=X_val.shape[0], timesteps=1, stateful=True, dropouts=dropouts )
    model2.load_weights( 'best_model_weights.h5' )
    return model2.test_on_batch( np.reshape(X_val,(X_val.shape[0],1,X_val.shape[1],X_val.shape[2],X_val.shape[3])),
        np.reshape(y_val, (y_val.shape[0],1,y_val.shape[1]) ) )

def runTests(args):
    arr1 = hparamsToArray( {} )
    print( "default hparams: {}".format( arr1 ) )
    dict1 = hparamsToDict( arr1 )
    arr2 = hparamsToArray( dict1 )
    if arr1 == arr2:
        print( "round trip worked" )
    else:
        print( "{}".format( arr2 ) )
    dict1["dropouts"] = "up"
    dropouts = [0.2,0.3,0.4,0.5,0.6]
    res = hparamsToArray(dict1)
    if dropouts == res[2]:
        print( "Dropouts with 'up' worked" )
    else:
        print( "Dropouts with 'up' did NOT work" )
    print( args.dirs )

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')
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

    images, y = loadData(args.dirs)
    input_dim = images[0].shape
    num_actions = len(y[0])
    num_samples = len(images)

    print( "Samples: {}   Input: {}  Output: {}".format( num_samples, input_dim, num_actions ) )
    print( "Shape of y: {}".format( y.shape ) )
    print( "Image 0 data: {} {}".format( np.min(images[0]), np.max(images[0]) ) )
    print( "Images: {}".format( images.shape ) )
    print( "Labels: {}".format( y.shape ) )

    # Get default params
    hparams = hparamsToDict( hparamsToArray( {} ) )
    vals = []
    histories = []
    count = 5
    verbose = 0 if (count > 1) else 1
    for i in range(count):
        val, his = fitLSTM( input_dim, images, y, verbose=verbose, **hparams )
        #val, his = fitFC( input_dim, images, y, verbose=verbose, **hparams )
        # Return all history from the fit methods and pickle
        vals.append(val)
        histories.append(his.history)

    with open("histories.pickle", 'wb') as f:
        pickle.dump( histories, f, pickle.HIGHEST_PROTOCOL)

    if count > 1:
        print( "Validation accuracy {} {} ({})".format( np.mean(vals), np.std(vals), vals ) )
