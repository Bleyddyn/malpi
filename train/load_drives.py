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
from augmentor import ImageAugmenter

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
    def __init__(self, filelist, image_size=(120,120), batch_size=32, shuffle=True, max_load=30000, auxName=None, images_only=False, augmenter=None, aug_factor=1 ):
        """ Input a list of drive directories.
            Pre-load each to count number of samples.
            load one file and use it to generate batches until we run out.
            load the next file, repeat
            Re-shuffle on each epoch end
        """
        self.files = filelist
        self.size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_load = max_load
        self.auxName = auxName
        self.images_only = images_only
        self.augment_factor = aug_factor
        self.augmenter = augmenter
        if aug_factor != 1 and augmenter is None:
            self.augmenter = ImageAugmenter( 1.0 / aug_factor )
        self.image_norm = False
        self.next_dir_index = 0
        self.images = []
        self.actions = []
        self.current_start = 0
        self.categorical = None
        self.input_dim = None
        self.num_actions = None
        self.batch_shape = (self.batch_size,) + self.size + (3,)
        for i in range(1,self.augment_factor):
            self.files.extend( self.files )
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

        if (sample_beg < len(self.images)) and (sample_end < len(self.images)):
            images = self.images[sample_beg:sample_end]
            actions = self.actions[sample_beg:sample_end]
            if self.images_only:
                if images.shape != self.batch_shape:
                    print( "Invalid batch shape (1): {} {} {} {}".format( images.shape, sample_beg, sample_end, prev_len) )
                return images
            else:
                return images, actions

        if sample_beg <= len(self.images):
            images = self.images[sample_beg:]
            actions = self.actions[sample_beg:]
            #sample_end = len(self.images) - sample_beg
            sample_end = self.batch_size - len(images)
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
            if self.images_only:
                if images.shape != self.batch_shape:
                    print( "Invalid batch shape (2): {} {} {} {}".format( images.shape, sample_beg, sample_end, prev_len) )
                return images
            else:
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
            if self.images_only:
                images = sklearn.utils.shuffle(images)
            else:
                images, actions = sklearn.utils.shuffle(images,actions)

        images = np.array(images).astype(np.float) / 255.0
        actions = np.array(actions)

        if self.image_norm:
            normalize_images(images)

        if self.augmenter is not None:
            self.augmenter(images)

        return images, actions

    def loadOneDrive( self, drive_dir, count_only=False ):
        actions = []
        if not self.images_only:
            if self.auxName is not None:
                aux = getAuxFromMeta( drive_dir, self.auxName )
                if aux is not None:
                    actions = loadOneAux( drive_dir, aux )
            if len(actions) == 0:
                actions_file = os.path.join( drive_dir, "image_actions.npy" )
                if os.path.exists(actions_file):
                    actions = np.load(actions_file)
                else:
                    actions_file = os.path.join( drive_dir, "image_actions.pickle" )
                    with open(actions_file,'r') as f:
                        actions = pickle.load(f)

            if len(actions) > 0:
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

                # Try training only the steering actions
                actions = actions[:,0]
                self.num_actions = 1

                # Need an option for this
                #if not self.categorical:
                #    actions = self.addActionDiff(actions)

                # Try scaling the action values up to get it to train
                if not self.categorical:
                    actions *= 100.0

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

        if count_only:
            return len(images)

        if not self.images_only and (len(images) != len(actions)):
            print( "Data mismatch: {}".format( drive_dir ) )
            print( "   images: {}".format( images.shape ) )
            print( "  actions: {}".format( actions.shape ) )

        return images, actions

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.files)
        self.images = []
        self.actions = []
        self.next_dir_index = 0
        self.current_start = 0
        self.images, self.actions = self.__load_next_max()

    def __count(self):
        count = 0
        for onefile in self.files:
            count += self.loadOneDrive( onefile, count_only=True)
        return count

    def addActionDiff(self, actions):
        diff = actions[:,0] - actions[:,1]
        #diff *= 10.0
        diff = np.reshape( diff, (diff.shape[0], 1) )
        actions = np.hstack( (actions, diff)  )
        return actions

def displayImages(images):
    import matplotlib.pyplot as plt

    n = 10
    plt.figure(figsize=(20, 4))
    plt.suptitle( "Sample Images", fontsize=16 )
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(images[i])
        #plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def runTests(args):

    n = 10
    gen = DriveDataGenerator(args.dirs, batch_size=n, shuffle=True, max_load=2000, images_only=True )

    print( "# samples: {}".format( gen.count ) )
    print( "# batches: {}".format( len(gen) ) )

    images = gen[0]

#    rmean = 92.93206363205326
#    gmean = 85.80540021330793
#    bmean = 54.14884297660608
#    rstd = 57.696159704394354
#    gstd = 53.739380109203445
#    bstd = 47.66536771313241
#
#    images[:,:,:,0] *= rstd
#    images[:,:,:,1] *= gstd
#    images[:,:,:,2] *= bstd
#    images[:,:,:,0] += rmean
#    images[:,:,:,1] += gmean
#    images[:,:,:,2] += bmean

    aug = ImageAugmenter( 0.5 )
    aug(images)
    displayImages(images)

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

    images_only = True
    gen = DriveDataGenerator(args.dirs, batch_size=32, shuffle=True, max_load=2000, images_only=images_only, aug_factor=2 )

    print( "# samples: {}".format( gen.count ) )
    print( "# batches: {}".format( len(gen) ) )

    if not images_only:
        _, actions = gen[0]
        print( "Actions: {}".format( actions.shape ) )
        print( "   mean: {}".format( np.mean(actions, axis=0) ) )
        print( "  stdev: {}".format( np.std(actions, axis=0) ) )
    #diff = actions[:,0] - actions[:,1]
    #print( "  Diffs: {}".format( diff.shape ) )
    #print( "   mean: {}".format( np.mean(diff, axis=0) ) )
    #print( "  stdev: {}".format( np.std(diff, axis=0) ) )
    #diff = np.reshape( diff, (diff.shape[0], 1) )
    #actions = np.hstack( (actions, diff)  )
    #print( "Actions: {}".format( actions.shape ) )

    #exit()

    for i in range(len(gen)):
        if images_only:
            images = gen[i]
            print( "Batch {}: {}".format( i, images.shape ), end='\r' )
        else:
            images, actions = gen[i]
        #print( "Batch {}: {} {}".format( i, images.shape, actions.shape ), end='\r' )
            print( "action[0]: {}".format( actions[0] ) )

        if i == 0:
            displayImages(images)
        sys.stdout.flush()
        time.sleep(0.1)
    print("")
