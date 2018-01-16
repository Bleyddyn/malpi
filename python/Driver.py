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

import keras

try:
    import config
except:
    print( "Failed to load config file config.py." )
    print( "Try copying config_empty.py to config.py and re-running." )
    exit()

def log( message, options ):
    logFileName = os.path.join(options.dir_ep, options.episode) + ".log"
    fmt = '%Y%m%d-%H%M%S.%f'
    datestr = datetime.datetime.now().strftime(fmt)

    with open(logFileName,'a') as outf:
        outf.write(datestr + ": " + message + "\n")

class Driver:
    def __init__(self, model_path, camera, controller):
        """ model_path: location of the saved keras model
            camera: A instance of PiVideoStream
            controller: DriveDaemon which handles creating the camera and sending motor commands
        """
        self.stopped = False
        self.model_path = model_path
        self.camera = camera
        self.controller = controller
        self.model = keras.models.load_model(model_path)
        self.image_delay = 0.1
        #self.embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
        self.embedding = [ "stop", "forward", "left", "right", "backward" ]

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

        (image,_) = self.camera.read()
        if image is not None:
# pre-process image
# pass image through model
# Choose action
# Pass action to controller
            image = self.pre_process(image)
            actions = self.model.predict_on_batch(image)
            action = self.softmax(actions)
            controller.do_action( self.embedding[action] )

    def _drive( self ):
        led.turnLEDOn( True, 11 )
        while not self.stopped:
            sleep(self.image_delay)
            self._step()
        led.turnAllLEDOn( False )

    def startDriving( self ):
        Thread(target=self._drive, args=()).start()

    def endDriving( self ):
        self.stop()

    def softmax(self, x):
      probs = np.exp(x - np.max(x))
      probs /= np.sum(probs )
      return probs

    def pre_process(self,  image, image_norm=True ):
        #image = np.array(image)
        image = images.astype(np.float) # / 255.0
        image = imresize(image, (120,120), interp='nearest' )
        image = image.reshape( 1, 120, 120, 3 )

        if image_norm:
            image[:,:,:,0] -= np.mean(image[:,:,:,0])
            image[:,:,:,1] -= np.mean(image[:,:,:,1])
            image[:,:,:,2] -= np.mean(image[:,:,:,2])
            image[:,:,:,0] /= np.std(image[:,:,:,0])
            image[:,:,:,1] /= np.std(image[:,:,:,1])
            image[:,:,:,2] /= np.std(image[:,:,:,2])

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
        options.episode = n.strftime('drive_%Y%m%d_%H%M%S') 

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
    with Driver( os.path.expanduser("~/drive/test1") ) as adrive:
        adrive.startDriving( options )
        sleep(20)
        adrive.endDriving()
