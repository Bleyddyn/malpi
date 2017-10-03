import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

setCPUCores( 4 )

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
#drive_dir = "/Users/Shared/Personal/ML/drive/drive_20170930_124550"
    dirs = [ "drive_20170930_124144", "drive_20170930_124230", "drive_20170930_124322", "drive_20170930_124407", "drive_20170930_124507", "drive_20170930_124550" ]
    dirs = [ "drive_20170930_124144", "drive_20170930_124230"]

    images = []
    actions = []

    for onedir in dirs:
        ddir = os.path.join("/Users/Shared/Personal/ML/drive", onedir )
        dimages, dactions = loadOneDrive( ddir )
        images.extend(dimages)
        actions.extend(dactions)

    images = np.array(images)
#images = np.array(data['images'])
#actions = data['image_actions']
    y = embedActions( actions )
    y = to_categorical( y, num_classes=5 )
    return images, y

def trainLSTM( images, y, epochs ):
    num_actions = len(y[0])
    input_dim = images[0].shape
    num_samples = len(images)
    model = model_keras.make_model_lstm( num_actions, input_dim, batch_size=1, timesteps=10 )
    timesteps = 10
    num_batches = images.shape[0] / timesteps
    losses = []
    accs = []
    for epoch in range(epochs):
        starts = range(0,images.shape[0],timesteps)
        np.random.shuffle(starts)
        epoch_losses = []
        epoch_accs = []
        for start in starts:
            end = start+timesteps
            t_b = timesteps
            if end >= num_samples:
                end = num_samples
                t_b = end - start
# Each of these is one batch of timesteps contiguous samples
            try:
                X = np.reshape(images[start:end,:], (1,t_b)+input_dim)
                y_batch = np.reshape( y[start:end,:], (1,t_b,num_actions))
            except ValueError as err:
                print( err )
                print( "Failed to reshape batch: {}:{}".format( start, end ) )
                print( "   From: {}".format( images[start:end,:].shape ) )
                print( "     To: {}".format( (1,timesteps)+input_dim ) )
                print( "   From: {}".format( y[start:end,:].shape ) )
                print( "     To: {}".format( (1,timesteps,num_actions) ) )
            (loss, acc) = model.train_on_batch( X, y_batch )
            model.reset_states() # Don't carry internal state between batches
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        mloss = np.mean(epoch_losses)
        macc = np.mean(epoch_accs)
        losses.append( mloss )
        accs.append( macc )
        print( "Epoch {}: loss: {}  acc: {}".format( epoch+1, mloss, macc ) )
    plotHistory( losses, accs, losses, accs )

if __name__ == "__main__":
    K.set_learning_phase(True)

    images, y = loadData()
    input_dim = images[0].shape
    num_actions = len(y[0])
    num_samples = len(images)
    epochs = 100
    recurrent = True
    print( "Samples: {}   Input: {}  Output: {}".format( num_samples, input_dim, num_actions ) )
    print( "Shape of y: {}".format( y.shape ) )

    if recurrent:
        trainLSTM( images, y, epochs )
    else:
        model = model_keras.make_model_test( num_actions, input_dim )
        history = model.fit( images, y, validation_split=0.25, epochs=epochs, callbacks=[SGDLearningRateTracker()] )

        #print( history.history.keys() )
        plotHistory( history.history['loss'], history.history['categorical_accuracy'],
                     history.history['val_loss'], history.history['val_categorical_accuracy'] )
