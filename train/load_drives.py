from __future__ import print_function
import os
import sys
import pickle
import time
import argparse

import numpy as np
import sklearn

from keras.utils import to_categorical
from keras.utils import Sequence
# https://keras.io/utils/#sequence

import malpiOptions
from load_aux import getAuxFromMeta, loadOneAux

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

def normalize_images( images, default=True ):
    if default:
        rmean = 92.93206363205326
        gmean = 85.80540021330793
        bmean = 54.14884297660608
        rstd = 57.696159704394354
        gstd = 53.739380109203445
        bstd = 47.66536771313241

        #print( "Default normalization" )
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd
    else:
        rmean = np.mean(images[:,:,:,0])
        gmean= np.mean(images[:,:,:,1])
        bmean= np.mean(images[:,:,:,2])
        rstd = np.std(images[:,:,:,0])
        gstd = np.std(images[:,:,:,1])
        bstd = np.std(images[:,:,:,2])
        print( "Image means: {}/{}/{}".format( rmean, gmean, bmean ) )
        print( "Image stds: {}/{}/{}".format( rstd, gstd, bstd ) )

# should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd

def embedActions( actions ):
    embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
    emb = []
    prev_act = 0
    for act in actions:
        try:
            if not act.startswith("speed"):
                prev_act = embedding[act]
                if prev_act is None:
                    print( "Invalid action: {}".format( act ) )
                    raise ValueError("Invalid action: " + str(act) )
                emb.append( embedding[act] )
            else:
                emb.append( prev_act )
        except Exception as ex:
            print( ex )
            print( act )
    return emb

class DriveDataGenerator(Sequence):
    """ Loads MaLPi drive data
        From: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html"""
    def __init__(self, filelist, image_size=(120,120), batch_size=32, shuffle=True, max_load=30000, auxName=None ):
        """ Input a list of drive directories.
            Pre-load each to count number of samples.
            load one file and use it to generate batches until we run out.
            load the next file, repeat
            Re-shuffle on each epoch end
        """
        'Initialization'
        self.files = filelist
        self.size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_load = max_load
        self.auxName = auxName
        self.image_norm = False
        self.next_dir_index = 0
        self.images = []
        self.actions = []
        self.current_start = 0
        self.categorical = None
        self.input_dim = None
        self.num_actions = None
        self.count = self.__count()
        self.on_epoch_end()

    def __len__(self):
        'The number of batches per epoch'
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, index):
        sample_beg = index * self.batch_size
        sample_beg -= self.current_start
        sample_end = sample_beg + self.batch_size
        #print( "getitem {} {}:{}".format( index, sample_beg, sample_end ) )

        if sample_end <= len(self.images):
            images = self.images[sample_beg:sample_end]
            actions = self.actions[sample_beg:sample_end]
            return images, actions

        if sample_beg <= len(self.images):
            images = self.images[sample_beg:]
            actions = self.actions[sample_beg:]
            sample_end = len(self.images) - sample_beg
            self.images, self.actions = self.__load_next_max()
            try:
                i2 = self.images[0:sample_end]
                images = np.append(images,i2,axis=0)
            except Exception as ex:
                print( ex )
                print( "images {} {}".format( images.shape, i2.shape ) )
                print( "{}".format( images ) )
            try:
                a2 = self.actions[0:sample_end]
                actions = np.append(actions,a2,axis=0)
            except Exception as ex:
                print( ex )
                print( "actions {} {}".format( actions.shape, a2.shape ) )
                print( "{}".format( actions ) )
            return images, actions
        print( "Invalid batch indexes: {} {}:{}".format( index, sample_beg, sample_end ) )

    def __load_next_max(self):

        self.current_start += len(self.images)

        images = []
        actions = []

        while len(images) <= self.max_load and self.next_dir_index < len(self.files):
            fname = self.files[self.next_dir_index]
            dimages, dactions = self.loadOneDrive( fname )
            images.extend(dimages)
            actions.extend(dactions)
            self.next_dir_index += 1

        if self.shuffle == True:
            images, actions = sklearn.utils.shuffle(images,actions)

        images = np.array(images)
        actions = np.array(actions)

        if self.image_norm:
            normalize_images(images)

        return images, actions

    def loadOneDrive( self, drive_dir, count_only=False ):
        actions = None
        if self.auxName is not None:
            aux = getAuxFromMeta( drive_dir, self.auxName )
            if aux is not None:
                actions = loadOneAux( drive_dir, aux )
        if actions is None:
            actions_file = os.path.join( drive_dir, "image_actions.npy" )
            if os.path.exists(actions_file):
                actions = np.load(actions_file)
            else:
                actions_file = os.path.join( drive_dir, "image_actions.pickle" )
                with open(actions_file,'r') as f:
                    actions = pickle.load(f)

        categorical = True
        if isinstance(actions[0], basestring):
            actions = embedActions( actions )
            actions = to_categorical( actions, num_classes=5 )
            categorical = True
        elif type(actions) == list:
            actions = np.array(actions).astype('float')
            categorical = False
        elif type(actions) == np.ndarray:
            actions = np.array(actions).astype('float')
            categorical = False
        else:
            print("Unknown actions format: {} {} as {}".format( type(actions), actions[0], type(actions[0]) ))

        if self.categorical is None:
            self.categorical = categorical
        elif self.categorical != categorical:
            print( "Mixed cat/non-cat action space: {}".format( drive_dir ) )

        if self.num_actions is None:
            self.num_actions = len(actions[0])

        if count_only:
            return len(actions)

        basename = "images_{}x{}".format( self.size[0], self.size[1] )
        im_file = os.path.join( drive_dir, basename+".npy" )
        if os.path.exists(im_file):
            images = np.load(im_file)
        else:
            im_file = os.path.join( drive_dir, basename+".pickle" )
            with open(im_file,'r') as f:
                images = pickle.load(f)

        if len(images) != len(actions):
            print( "Data mismatch: {}".format( drive_dir ) )
            print( "   images: {}".format( images.shape ) )
            print( "  actions: {}".format( actions.shape ) )

        return images, actions

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.files)
        self.next_dir_index = 0
        self.current_start = 0
        self.images, self.actions = self.__load_next_max()

    def __count(self):
        count = 0
        for onefile in self.files:
            count += self.loadOneDrive( onefile, count_only=True)
        return count

def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Test data loader.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    malpiOptions.addMalpiOptions( parser )
    args = parser.parse_args()
    malpiOptions.preprocessOptions(args)

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    gen = DriveDataGenerator(args.dirs, batch_size=32, shuffle=True, max_load=2000, auxName="Continuous" )

    print( "# samples: {}".format( gen.count ) )
    print( "# batches: {}".format( len(gen) ) )
    for i in range(len(gen)):
        images, actions = gen[i]
        #print( "Batch {}: {} {}".format( i, images.shape, actions.shape ), end='\r' )
        print( "action[0]: {}".format( actions[0] ) )
        sys.stdout.flush()
        time.sleep(0.1)
    print("")
