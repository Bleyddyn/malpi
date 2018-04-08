import os
from time import time
from time import sleep
import datetime
from optparse import OptionParser
import io
import shutil
from subprocess import Popen, PIPE, STDOUT
import pickle
import socket
from threading import Thread, Lock

import numpy as np

from accelerometer import accelerometer
#from PiVideoStream import PiVideoStream
import led

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

class DriveRecorder:
    def __init__(self, drive_dir, video_path=None, camera=None, image_delay=None, drive_name=None):
        """ drive_dir: directory where all data will be saved
            video_path: location of the saved video file, e.g. /var/ramdrive/video.h264. Will be moved into drive_dir
            camera: A instance of PiVideoStream
            image_delay: time between capturing images, in seconds
        """
        self.stopped = False
        self.drive_dir = drive_dir
        self.video_path = video_path
        self.camera = camera
        self.image_delay = image_delay
        if drive_name is None:
            drive_name = "Manual Drive"
        self.drive_name = drive_name
        self.accel = None

        self.images = []
        self.image_index = 1
        self.max_images = 10
        self.image_times = []
        self.image_actions = []
        self.actions = ["stop"]
        self.action_times = []

    def setVideoFilename(filename):
        self.video_path = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # I'm not 100% sure this will result in everything being closed. Best to also call stop().
        self.stopped = True

    def stop(self,signum=0,frame=0):
        # indicate that the thread should be stopped
        self.stopped = True
        if self.accel:
            self.accel.terminate()
            self.accel = None
        if self.camera:
            self.camera.stopVideo()
        led.turnAllLEDOn( False )

    def addAction(self,action):
        self.actions.append(action)
        self.action_times.append(time())

    def addImage(self,image,format='numpy', last_action=None):
        # This method is no longer being used
        if format == 'jpeg':
            #convert to numpy
            pass
        self.images.append(image)
        self.image_times.append(time())
        if last_action is not None:
            self.image_actions.append(last_action)

    def captureImage( self ):
        while not self.stopped:
            sleep(self.image_delay)
            if self.camera and not self.stopped:
                (full,_) = self.camera.read()
                if full is not None:
                    self.images.append( full )
                    self.image_times.append( time() )
                    self.image_actions.append( self.actions[-1] )
                if len(self.images) >= self.max_images:
                    self.saveImages()
            else:
                return

    def saveImages( self ):
        # Save current image array and empty it.
        images_filename = os.path.join( self.drive_dir, "images_{}.pickle".format(self.image_index) )
        with open(images_filename, 'wb') as f:
            pickle.dump( self.images, f, pickle.HIGHEST_PROTOCOL)
        self.images = []
        self.image_index += 1

    def startDriving( self ):
        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir)

        self.accel_path = os.path.join( self.drive_dir, "accel.pkl" )
        #self.accel = Popen([config.directories['accelerometer_script'], self.accel_path])
        if self.camera:
            if self.image_delay and self.image_delay > 0.0:
                Thread(target=self.captureImage, args=()).start()
            self.camera.startVideo(self.video_path)
        led.turnLEDOn( True, 11 )


    def endDriving( self ):
        self.stop()
        accel_data = []
        sleep(1) 
        try:
            with open(self.accel_path, 'rb') as f:
                accel_data = pickle.load( f )
        except:
            pass
        self.accel_path = None

        led.turnLEDOn( True, 12 )
        self.saveImages()
        pkg = self.packageImages( self.images, self.image_times, self.actions, self.action_times, accel_data, self.image_actions )
        images_filename = os.path.join( self.drive_dir, "drive.pickle" )
        with open(images_filename, 'wb') as f:
            pickle.dump( pkg, f, pickle.HIGHEST_PROTOCOL)
        if os.path.exists(self.video_path):
            shutil.move( self.video_path, os.path.join( self.drive_dir, "drive_video.h264") )
        led.turnAllLEDOn( False )

    def packageImages(self, images, image_times, actions, action_times, accel_data, image_actions ):
        image_pkg = { }
        image_pkg["date"] = datetime.datetime.now()
        image_pkg["model"] = self.drive_name
        image_pkg["actions"] = actions
        image_pkg["action_times"] = action_times
        image_pkg["accelerometer"] = accel_data
        image_pkg["images"] = images
        image_pkg["image_times"] = image_times
        image_pkg["image_actions"] = image_actions
        return image_pkg

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
    with DriveRecorder( os.path.expanduser("~/drive/test1") ) as adrive:
        adrive.startDriving( options )
        sleep(20)
