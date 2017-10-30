import os
import pickle
from time import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model_keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
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

def plotHistory( loss, acc, val_loss, val_acc ):
    #['val_categorical_accuracy', 'loss', 'categorical_accuracy', 'val_loss']

    # summarize history for accuracy
    plt.figure(1,figsize=(10, 15), dpi=80)
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

def loadData( dirs ):
    images = []
    actions = []

    for onedir in dirs:
        if len(onedir) > 0:
            #ddir = os.path.join("./drive", onedir )
            dimages, dactions = loadOneDrive( onedir )
            images.extend(dimages)
            actions.extend(dactions)

    images = np.array(images)
    images = images.astype(np.float) / 255.0
#images = np.array(data['images'])
#actions = data['image_actions']
    y = embedActions( actions )
    y = to_categorical( y, num_classes=5 )
    return images, y

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def hparamsToArray( hparams ):
    out = []
    out.append( hparams.get( "timesteps", 10 ) )
    out.append( hparams.get( "l2_reg", 0.005 ) )
    out.append( hparams.get( "dropouts", [0.25,0.25,0.25,0.25,0.25] ) )
    out.append( hparams.get( "learning_rate", 0.003 ) )
    out.append( hparams.get( "validation_split", 0.15 ) )
    out.append( hparams.get( "batch_size", 5 ) )
    out.append( hparams.get( "optimizer", "RMSprop" ) )
    out.append( hparams.get( "epochs", 40 ) )
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
    if verbose:
        save_chk = ModelCheckpoint("weights_{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks = [save_chk]

    model = model_keras.make_model_lstm_fit( num_actions, input_dim, timesteps=timesteps, stateful=False, dropouts=dropouts )

    history = model.fit( images, y, validation_split=validation_split, epochs=epochs, verbose=verbose, batch_size=batch_size, shuffle=False, callbacks=callbacks )

    if verbose:
        model.save( 'best_model.h5' )
        model.save_weights('best_model_weights.h5')
        plotHistory( history.history['loss'], history.history['categorical_accuracy'],
                     history.history['val_loss'], history.history['val_categorical_accuracy'] )

        if X_val is not None and y_val is not None:
            (val_loss, val_acc) = evaluate( num_actions, input_dim, X_val, y_val, dropouts=dropouts )
            print( "Final Validation loss/acc: {}  {}".format( val_loss, val_acc) )

    running = runningMean(history.history['val_categorical_accuracy'], 5)
    max_running = np.max( running )
    print( "Max validation (rmean=5, at {}): {}".format( np.argmax(running), max_running ) )
    #print( "  val_accuracy: {}".format( history.history['val_categorical_accuracy'] ) )

    return max_running

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

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('--file', help='File with one directory per line')
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
    timesteps = 10

#    hold_out = (num_samples % timesteps) + (5 * timesteps)
#    num_samples = num_samples - hold_out
#    X_val = images[num_samples:,:]
#    y_val = y[num_samples:,:]
#    images = images[0:num_samples,:]
#    images = np.reshape( images, (num_samples/timesteps, timesteps) + input_dim )
#    y = y[0:num_samples,:]
#    y = np.reshape( y, (num_samples/timesteps, timesteps, num_actions) )
#    epochs = 40

    print( "Samples: {}   Input: {}  Output: {}".format( num_samples, input_dim, num_actions ) )
    print( "Shape of y: {}".format( y.shape ) )
    print( "Image 0 data: {} {}".format( np.min(images[0]), np.max(images[0]) ) )
    print( "Images: {}".format( images.shape ) )
    print( "Labels: {}".format( y.shape ) )

    # Get default params
    hparams = hparamsToDict( hparamsToArray( {} ) )
    vals = []
    count = 1
    verbose = 0 if (count > 1) else 1
    for i in range(count):
        val = fitLSTM( input_dim, images, y, verbose=verbose, **hparams )
        vals.append(val)

    if count > 1:
        print( "Validation accuracy {} {} ({})".format( np.mean(vals), np.std(vals), vals ) )
