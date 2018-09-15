import os
from time import time
from time import sleep
import datetime
from optparse import OptionParser
import pickle
from threading import Thread, Lock

import numpy as np
from scipy.misc import imresize

import led

#import keras
import keras.backend as K
import model_keras
import tensorflow as tf

try:
    import config
except:
    print( "Driver.py failed to load config file config.py." )
    print( "Try copying config_empty.py to config.py and re-running." )
    exit()

def log( message, options ):
    logFileName = os.path.join(options.dir_ep, options.episode) + ".log"
    fmt = '%Y%m%d-%H%M%S.%f'
    datestr = datetime.datetime.now().strftime(fmt)

    with open(logFileName,'a') as outf:
        outf.write(datestr + ": " + message + "\n")

class Driver:
    def __init__(self, model_path, camera, controller, model_name=None):
        """ model_path: location of the saved keras model
            camera: A instance of PiVideoStream
            controller: DriveDaemon which handles creating the camera and sending motor commands
        """
        K.set_learning_phase(False)

        self.stopped = False
        self.model_path = model_path
        self.camera = camera
        self.controller = controller
        #self.embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
        self.embedding = [ "stop", "forward", "left", "right", "backward" ]
        self.image_delay = 0.01
        led.turnLEDOn( True, 11 )
        led.turnLEDOn( True, 13 )
        self.isRNN = False
        self.continuous = False
        if self.isRNN:
            self.model = model_keras.make_model_lstm( len(self.embedding), (120,120,3), batch_size=1, timesteps=1, stateful=True, dropouts=[0.25,0.25,0.25,0.25,0.25] )
        else:
            #self.model = model_keras.make_model_test( len(self.embedding), (120,120,3), dropouts=[0.25,0.25,0.25,0.25,0.25] )
            if self.continuous:
                self.model = model_keras.make_model_fc( 3, (120,120,3), dkconv=True, dropouts=[0.25,0.25,0.25,0.25,0.25], categorical=False )
                #self.model = model_keras.read_model( model_path, model_name )
            else:
                self.model = model_keras.read_model( model_path, model_name )
                #self.model = model_keras.make_model_fc( len(self.embedding), (120,120,3), dkconv=True, dropouts=[0.25,0.25,0.25,0.25,0.25] )
        self.model.load_weights( os.path.join( model_path, model_name+"_weights.h5" ) )
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        #self.graph = K.get_session().graph # This should work, too
        led.turnAllLEDOn( False )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # I'm not 100% sure this will result in everything being closed. Best to also call stop().
        self.stopped = True

    def stop(self,signum=0,frame=0):
        # indicate that the thread should be stopped
        self.stopped = True
        led.turnAllLEDOn( False )

    def _step(self):
        if self.stopped:
            return
        (image,pre_image) = self.camera.read()
        image = pre_image
        if image is not None:
            t2 = time()
            image = self.pre_process(image)
            t3 = time()
            actions = self.model.predict_on_batch(image)
            if self.isRNN:
                actions = actions[0][0]
            else:
                actions = actions[0]
            t4 = time()
            #print( "Actions: {}".format( actions ) )
            #if action == 0:
            #    action = np.argmax( actions[1:] ) + 1
            #    #print( "skipping stop action" )
            if self.continuous:
                action = 'throttles {} {}'.format( actions[0], actions[1] )
                self.controller.do_action( action )
            else:
                action = np.argmax(actions) # No exploration, just choose the best
                self.controller.do_action( self.embedding[action] )
            #print( "Times; {} {}".format( t3-t2, t4-t3 ) )

    def _drive( self ):
        led.turnLEDOn( True, 11 )
        with self.graph.as_default():
            while not self.stopped:
                sleep(self.image_delay)
                self._step()
        self.controller.do_action( "stop" )
        led.turnAllLEDOn( False )

    def startDriving( self ):
        self.stopped = False
        Thread(target=self._drive, args=()).start()

    def endDriving( self ):
        self.stop()

    def softmax(self, x):
      probs = np.exp(x - np.max(x))
      probs /= np.sum(probs )
      return probs

    def pre_process(self,  image, image_norm=True ):
        image = image.astype(np.float) # / 255.0
        if image.shape[0] > 120 or image.shape[1] > 120:
            image = imresize(image, (120,120), interp='nearest' ) # This is slow, 0.3 - 0.4 seconds

        if image_norm:
            image[:,:,0] -= np.mean(image[:,:,0])
            image[:,:,1] -= np.mean(image[:,:,1])
            image[:,:,2] -= np.mean(image[:,:,2])
            image[:,:,0] /= np.std(image[:,:,0])
            image[:,:,1] /= np.std(image[:,:,1])
            image[:,:,2] /= np.std(image[:,:,2])

        if self.isRNN:
            image = image.reshape( 1, 1, 120, 120, 3 )
        else:
            image = image.reshape( 1, 120, 120, 3 )

        return image

def getOptions():
    usage = "Usage: python ./drive.py [options]"
    parser = OptionParser()
    parser.add_option("-e","--episode", help="Episode Name. Used for episode related file names. Defaults to date/time.");
    parser.add_option("--dir_ep", help="Directory for saving all episode related files. Defaults to episode name.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--video", action="store_true", default=False, help="Record video during an episode.");
    parser.add_option("-s","--steps", type="int", default=300, help="Number of steps to run. Defaults to 300.");

    (options, args) = parser.parse_args()

    if not options.episode:
        n = datetime.datetime.now()
        options.episode = n.strftime('%Y%m%d_%H%M%S.drive') 

    if not options.dir_ep:
        options.dir_ep = options.episode
    else:
        options.dir_ep = os.path.join( options.dir_ep, options.episode )

    options.dir_ep = os.path.expanduser(options.dir_ep)

    # Hard code this for now
    options.im_format = "numpy"

    return (options, args)

def test(options):
    cwd = os.getcwd()
    print( cwd )
    actions_path = os.path.join( options.dir_ep, options.episode+"_actions.text" )
    print( actions_path )
    actions_path = os.path.abspath(actions_path)
    print( actions_path )
    exit()

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.test_only:
        test(options)

    #def __init__(self, drive_dir, video_path=None, camera=None, image_delay=None):
    #with Driver( os.path.expanduser("~/models/default.h5"), None, None ) as adrive:
    with Driver( os.path.expanduser("~/models"), None, None, model_name="default" ) as adrive:
        adrive.startDriving()
        sleep(20)
        adrive.endDriving()
