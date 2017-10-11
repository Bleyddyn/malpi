import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model_keras
from keras.utils import to_categorical
from keras.callbacks import Callback
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

def loadData():
    dirs = [ "drive_20170930_124144", "drive_20170930_124230", "drive_20170930_124322", "drive_20170930_124407", "drive_20170930_124507", "drive_20170930_124550" ]
    #dirs = [ "drive_20170930_124144", "drive_20170930_124230"]

    images = []
    actions = []

    for onedir in dirs:
        ddir = os.path.join("./drive", onedir )
        dimages, dactions = loadOneDrive( ddir )
        images.extend(dimages)
        actions.extend(dactions)

    images = np.array(images)
    images = images.astype(np.float) / 255.0
#images = np.array(data['images'])
#actions = data['image_actions']
    y = embedActions( actions )
    y = to_categorical( y, num_classes=5 )
    return images, y

def evaluate( num_actions, input_dim, X_val, y_val ):
    model2 = model_keras.make_model_lstm( num_actions, input_dim, batch_size=X_val.shape[0], timesteps=1 )
    model2.load_weights( 'best_model_weights.h5' )
    return model2.test_on_batch( np.reshape(X_val,(X_val.shape[0],1,X_val.shape[1],X_val.shape[2],X_val.shape[3])),
        np.reshape(y_val, (y_val.shape[0],1,y_val.shape[1]) ) )

def trainLSTM( images, y, epochs, batch_size, timesteps=10 ):
    num_actions = len(y[0])
    input_dim = images[0].shape
    num_samples = len(images)
    model = model_keras.make_model_lstm( num_actions, input_dim, batch_size=batch_size, timesteps=timesteps )
    printLearningRate(model)
    bt_size = batch_size * timesteps
    num_batches = images.shape[0] / bt_size
    extra = num_samples - (bt_size * num_batches)
    extra += bt_size
    last_start = images.shape[0] - (2 * extra)
    X_val = images[-extra:,:]
    y_val = y[-extra:,:]
    print( "Validation {}: {}".format( extra, X_val.shape ) )
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    best_accuracy = -10.0
    full_start = time()
    for epoch in range(epochs):
        epoch_start = time()
        starts = range(0,last_start,bt_size)
        np.random.shuffle(starts)
        epoch_losses = []
        epoch_accs = []
        for start in starts:
            end = start+bt_size
            t_b = bt_size
            if end >= num_samples:
                end = num_samples
                t_b = end - start
# Each of these is one batch of timesteps contiguous samples
            try:
                X = np.reshape(images[start:end,:], (batch_size,timesteps)+input_dim)
                y_batch = np.reshape( y[start:end,:], (batch_size,timesteps,num_actions))
            except ValueError as err:
                print( err )
                print( "Failed to reshape batch: {}:{}".format( start, end ) )
                print( "   From: {}".format( images[start:end,:].shape ) )
                print( "     To: {}".format( (batch_size,timesteps)+input_dim ) )
                print( "   From: {}".format( y[start:end,:].shape ) )
                print( "     To: {}".format( (batch_size,timesteps,num_actions) ) )
            (loss, acc) = model.train_on_batch( X, y_batch )
            model.reset_states() # Don't carry internal state between batches
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            if acc > best_accuracy:
                model.save( 'best_model.h5' )
                model.save_weights('best_model_weights.h5')
                best_accuracy = acc
        mloss = np.mean(epoch_losses)
        macc = np.mean(epoch_accs)
        losses.append( mloss )
        accs.append( macc )
        print( "Epoch {}: loss: {}  acc: {}".format( epoch+1, mloss, macc ) )
        (val_loss, val_acc) = evaluate( num_actions, input_dim, X_val, y_val )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print( "     val: loss: {}  acc: {}".format( val_loss, val_acc ) )
        printLearningRate(model)
        print( " seconds: {}".format( time() - epoch_start ) )
    print( " Total seconds: {}".format( time() - full_start ) )
    plotHistory( losses, accs, val_losses, val_accs )

if __name__ == "__main__":
    K.set_learning_phase(True)
    setCPUCores( 4 )

    images, y = loadData()
    input_dim = images[0].shape
    num_actions = len(y[0])
    num_samples = len(images)
    epochs = 100
    model_type = "lstm_batch"
    #model_type = "recurrent"
    #model_type = "forward"
    print( "Samples: {}   Input: {}  Output: {}".format( num_samples, input_dim, num_actions ) )
    print( "Shape of y: {}".format( y.shape ) )
    print( "Model Type: {}".format( model_type ) )
    print( "Image 0 data: {} {}".format( np.min(images[0]), np.max(images[0]) ) )
    print( "Images: {}".format( images.shape ) )
    print( "Labels: {}".format( y.shape ) )
    if model_type == "recurrent":
        trainLSTM( images, y, epochs, batch_size=1 )
    elif model_type == "lstm_batch":
        trainLSTM( images, y, epochs, batch_size=5 )
    else:
        model = model_keras.make_model_test( num_actions, input_dim )
        history = model.fit( images, y, validation_split=0.25, epochs=epochs, callbacks=[SGDLearningRateTracker()] )

        #print( history.history.keys() )
        plotHistory( history.history['loss'], history.history['categorical_accuracy'],
                     history.history['val_loss'], history.history['val_categorical_accuracy'] )
