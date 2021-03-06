from __future__ import print_function
import os
import sys
import pickle
from time import time
import argparse

import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model_keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import keras.backend as K

import malpiOptions
import experiment
import notify
from load_aux import loadAuxData
from load_drives import DriveDataGenerator

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

def describeDriveData( data ):
    print( data.keys() )
    for key, value in data.items():
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
            if prev_act is None:
                print( "Invalid action: {}".format( act ) )
                raise ValueError("Invalid action: " + str(act) )
            emb.append( embedding[act] )
        else:
            emb.append( prev_act )
    return emb

def setCPUCores( cores ):
    # Actual device_count seems to have less effect than number of threads
    config = tf.ConfigProto(intra_op_parallelism_threads=cores, inter_op_parallelism_threads=cores,
                            allow_soft_placement=True, device_count = {'CPU': cores})
    set_session(tf.Session(config=config))

def loadOneDrive( drive_dir, size=(120,120) ):
    actions_file = os.path.join( drive_dir, "image_actions.npy" )
    if os.path.exists(actions_file):
        actions = np.load(actions_file)
    else:
        actions_file = os.path.join( drive_dir, "image_actions.pickle" )
        with open(actions_file,'r') as f:
            actions = pickle.load(f)

    basename = "images_{}x{}".format( size[0], size[1] )
    im_file = os.path.join( drive_dir, basename+".npy" )
    if os.path.exists(im_file):
        images = np.load(im_file)
    else:
        im_file = os.path.join( drive_dir, basename+".pickle" )
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
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * tf.cast(optimizer.iterations, tf.float32))))
        print('\nLR: {:.6f}'.format(lr))

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

def loadData( dirs, size=(120,120), image_norm=True ):
    images = []
    actions = []

    count = 1
    for onedir in dirs:
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir, size=size )
            if dimages.shape[0] != dactions.shape[0]:
                print( "Data mismatch in {}: {} != {}".format( onedir, dimages.shape[0], dactions.shape[0] ) )
            dimages = dimages.astype(np.float)
            images.extend(dimages)
            actions.extend(dactions)
            print( "Loading {} of {}: {} total samples".format( count, len(dirs), len(images) ), end='\r' )
            sys.stdout.flush()
            count += 1

    print("")
    images = np.array(images)
    #images = images.astype(np.float) # / 255.0

    if image_norm:
        rmean = 92.93206363205326
        gmean = 85.80540021330793
        bmean = 54.14884297660608
        rstd = 57.696159704394354
        gstd = 53.739380109203445
        bstd = 47.66536771313241

        print( "Default normalization" )
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd

#        rmean = np.mean(images[:,:,:,0])
#        gmean= np.mean(images[:,:,:,1])
#        bmean= np.mean(images[:,:,:,2])
#        rstd = np.std(images[:,:,:,0])
#        gstd = np.std(images[:,:,:,1])
#        bstd = np.std(images[:,:,:,2])
#        print( "Image means: {}/{}/{}".format( rmean, gmean, bmean ) )
#        print( "Image stds: {}/{}/{}".format( rstd, gstd, bstd ) )
#
## should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
#        images[:,:,:,0] -= rmean
#        images[:,:,:,1] -= gmean
#        images[:,:,:,2] -= bmean
#        images[:,:,:,0] /= rstd
#        images[:,:,:,1] /= gstd
#        images[:,:,:,2] /= bstd

    categorical = True
    if isinstance(actions[0], basestring):
        actions = np.array(actions)
        actions = actions.astype('str')
        actions = embedActions( actions )
        actions = to_categorical( actions, num_classes=5 )
        categorical = True
    elif type(actions) == list:
        actions = np.array(actions)
        categorical = False
    else:
        print("Unknown actions format: {} {} as {}".format( type(actions), actions[0], type(actions[0]) ))

    return images, actions, categorical

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

def fitFC( input_dim, num_actions, gen, val, val_set=None, verbose=1, dkconv=False, early_stop=False, categorical=True, pre_model=None, epochs=40,
           timesteps=None, l2_reg=0.005, dropouts=[0.25,0.25,0.25,0.25,0.25], learning_rate=0.003, validation_split=0.15,
           batch_size=32, optimizer="RMSprop" ):
    callbacks = None

    callbacks = []

    if early_stop:
        callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0005, patience=5, verbose=verbose, mode='auto'))

    optimizer = makeOptimizer( optimizer, learning_rate )
    
    model = model_keras.make_model_fc( num_actions, input_dim, dkconv=dkconv, optimizer=optimizer, dropouts=dropouts,
                categorical=categorical, pre_model=pre_model )

    history = model.fit_generator( gen, validation_data=val, epochs=epochs, verbose=verbose, callbacks=callbacks, shuffle=False )

    if categorical:
        rm = 'val_categorical_accuracy'
    else:
        #rm = 'val_mean_squared_error'
        rm = 'val_loss'
    running = runningMean(history.history[rm], 5)
    if categorical:
        best_running = np.max( running )
        print( "Max validation (rmean=5, at {}): {}".format( np.argmax(running), best_running ) )
    else:
        best_running = np.min( running )
        print( "Min loss (rmean=5, at {}): {}".format( np.argmin(running), best_running ) )

    return (best_running, history, model)

def fitLSTM( input_dim, images, y, verbose=1, dkconv=False, early_stop=False, epochs=40, timesteps=10, l2_reg=0.005, dropouts=[0.25,0.25,0.25,0.25,0.25],
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

    callbacks = []
#    if verbose:
#        save_chk = ModelCheckpoint("weights_{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#        callbacks.append(save_chk)

    if early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=verbose, mode='auto'))

    optimizer = makeOptimizer( optimizer, learning_rate )

    model = model_keras.make_model_lstm_fit( num_actions, input_dim, dkconv=dkconv, timesteps=timesteps, stateful=False, dropouts=dropouts, optimizer=optimizer )

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

    return (max_running, history, model)

def evaluate( num_actions, input_dim, X_val, y_val, dropouts=[0.25,0.25,0.25,0.25,0.25] ):
    model2 = model_keras.make_model_lstm( num_actions, input_dim, batch_size=X_val.shape[0], timesteps=1, stateful=True, dropouts=dropouts )
    model2.load_weights( 'best_model_weights.h5' )
    return model2.test_on_batch( np.reshape(X_val,(X_val.shape[0],1,X_val.shape[1],X_val.shape[2],X_val.shape[3])),
        np.reshape(y_val, (y_val.shape[0],1,y_val.shape[1]) ) )

def evaluateRandom( input_dim, images, y, args, hparams ):
    num_actions = len(y[0])

    dropouts=hparams['dropouts']
    if args.fc:
        model2 = model_keras.make_model_fc( num_actions, input_dim, dkconv=args.dk, dropouts=dropouts )
        loss, val = model2.test_on_batch( images, y )
    else:
        num_samples = len(images)
        timesteps = 1
        hold_out = (num_samples % timesteps) + (5 * timesteps)
        num_samples = num_samples - hold_out
        X_val = images[num_samples:,:]
        y_val = y[num_samples:,:]
        images = images[0:num_samples,:]
        images = np.reshape( images, (num_samples/timesteps, timesteps) + input_dim )
        y = y[0:num_samples,:]
        y = np.reshape( y, (num_samples/timesteps, timesteps, num_actions) )
        model2 = model_keras.make_model_lstm_fit( num_actions, input_dim, dkconv=args.dk, timesteps=1, stateful=False, dropouts=dropouts )
        loss, val = model2.test_on_batch( images, y )

    print( val )
    his = None
    return (val, his, model2)

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

    images, y, cat = loadData(args.dirs)
    print( "Images: {}".format( len(images) ) )
    print( "Actions: {}".format( len(y) ) )
    print( "Actions: {}".format( y[0:5] ) )

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    #parser.add_argument('-f', '--file', help='File with one directory per line')
    parser.add_argument('--fc', action="store_true", default=False, help='Train a model with a fully connected layer (no RNN)')
    parser.add_argument('--dk', action="store_true", default=False, help='Train a model with DonkeyCar style Convolution layers')
    parser.add_argument('--early', action="store_true", default=False, help='Stop training early if learning plateaus')
    parser.add_argument('--runs', type=int, default=1, help='How many runs to train')
    parser.add_argument('--name', help='Display name for this training experiment')
    parser.add_argument('--aux', default=None, help='Use this auxiliary data in place of standard actions')
    parser.add_argument('--model', default=None, help='A file containing weights to pre-load the model')
    parser.add_argument('--val', default=None, help='A file with a list of directories to be used for validation')
    parser.add_argument('--val_split', type=float, default=0.2, help='Percent validation split')
    parser.add_argument('--notify', help='Email address to notify when the training is finished')
    #parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
    parser.add_argument('--random', action="store_true", default=False, help='Test an untrained model, then exit')
    parser.add_argument('--aug', type=int, default=None, help='Augment images by this factor. 2 = twice as many images, half of which are altered')

    malpiOptions.addMalpiOptions( parser )
    args = parser.parse_args()
    malpiOptions.preprocessOptions(args)

    if len(args.dirs) == 0 and not args.test_only:
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    if args.val is None:
        last = int(len(args.dirs) * args.val_split)
        np.random.shuffle(args.dirs)
        test = args.dirs[:last]
        train = args.dirs[last:]
        args.dirs = train
        args.val = test

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    K.set_learning_phase(True)
    setCPUCores( 4 )

    image_size = (120,120)
    input_dim = (120,120,3)
    gen = DriveDataGenerator(args.dirs, image_size=image_size, batch_size=60, shuffle=True, max_load=10000, auxName=args.aux, aug_factor=args.aug, binned=True )
    val = DriveDataGenerator(args.val, image_size=image_size, batch_size=60, shuffle=True, max_load=10000, auxName=args.aux, binned=True )

    num_actions = gen.num_actions
    num_samples = gen.count
    cat = gen.categorical

    print( "Samples: {}   Input: {}  Output: {}".format( num_samples, input_dim, num_actions ) )
    print( "Val Samples: {}".format( val.count ) )
    if not cat:
        print( "   Continuous actions" )

    if args.fc:
        print( "Fully connected layer, no RNN" )
    else:
        print( "RNN layer" )
    if args.dk:
        print( "DonkeyCar style convolution layers (5)" )
    else:
        print( "DeepMind style convolution layers (3)" )

    # Get default params
    #hparams = {'epochs': 100, 'optimizer': 'Adam', 'learning_rate': 0.0001, 'dropouts': 'up', 'batch_size': 10.0, 'l2_reg': 5.2e-07}
    hparams = {'epochs': 40, 'optimizer': 'Adam', 'learning_rate': 0.0005897214669321487, 'dropouts': 'up', 'batch_size': 60.0, 'l2_reg': 0.0074109846420101}
    hparams = hparamsToDict( hparamsToArray( hparams ) )
    if not args.random and args.name is not None:
        expMeta = experiment.Meta(args.name, args, num_samples=num_samples, input_dim=input_dim, num_actions=num_actions, hparams=hparams)

    best_model = None
    best_val = 0.0 if cat else 1000.0
    vals = []
    histories = []
    count = args.runs
    verbose = 0 if (count > 1) else 1
    for i in range(count):
        if args.random:
            val, his, model = evaluateRandom( input_dim, images, y, args, hparams )
        elif args.fc:
            val, his, model = fitFC( input_dim, num_actions, gen, val, verbose=verbose, dkconv=args.dk, early_stop=args.early, categorical=cat, pre_model=args.model, **hparams )
        else:
            val, his, model = fitLSTM( input_dim, images, y, verbose=verbose, dkconv=args.dk, early_stop=args.early, **hparams )
        # Return all history from the fit methods and pickle
        vals.append(val)
        if his is not None:
            histories.append(his.history)
        if (cat and (val > best_val)) or (not cat and (val < best_val)) :
            best_val = val
            best_model = model

    if not args.random and args.name is not None:
        expMeta.writeAfter(model=model, histories=histories, results={'vals': vals}, saveModel=True)

#    with open("histories.pickle", 'wb') as f:
#        pickle.dump( histories, f, pickle.HIGHEST_PROTOCOL)

    if count > 1:
        msg = "Validation accuracy {} {} ({})".format( np.mean(vals), np.std(vals), vals )
        print( msg )
    else:
        msg = "Validation accuracy {}".format( vals )

    msg2 = ""
    if args.name is not None:
        msg2 = "Model " + args.name

    notify.notify( "Training complete", subTitle=msg2, message=msg, email_to=args.notify )
