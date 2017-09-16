import os
from time import time
from time import sleep
import datetime
from optparse import OptionParser
import cStringIO
import io
import shutil
from subprocess import Popen, PIPE, STDOUT
import signal
import pickle
import socket

import numpy as np
from scipy import ndimage
from scipy import misc

from accelerometer import accelerometer
from PiVideoStream import PiVideoStream

try:
    import config
except:
    print "Failed to load config file config.py."
    print "Try copying config_empty.py to config.py and re-running."
    exit()

def log( message, options ):
    logFileName = os.path.join(options.dir_ep, options.episode) + ".log"
    fmt = '%Y%m%d-%H%M%S.%f'
    datestr = datetime.datetime.now().strftime(fmt)

    with open(logFileName,'a') as outf:
        outf.write(datestr + ": " + message + "\n")

def getOneTestImage(imsize):
    image = ndimage.imread('test_data/image.jpeg')
#image.shape (480, 720, 3)
    image = image.transpose(2,1,0)
    # shape = (3, 720, 480)
    min = (720 - 480) / 2
    image = image[:,min:min+480,:]
    image = misc.imresize(image,(imsize,imsize))
    # shape = (3, 480, 480)
    image = image.reshape(1,3,imsize,imsize)
    image = image.astype(np.float32)
    return image

def crop_center( img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocessOneImage(image, imsize):
#image.shape (480, 720, 3)
    image = image.transpose(2,1,0)
#    width = image.shape[1]
    # shape = (3, 720, 480)
#    if width > 480:
#        min = (width - 480) / 2
#        image = image[:,min:min+480,:]
#    image = misc.imresize(image,(imsize,imsize))
    # shape = (3, imsize, imsize)
    image = image.reshape(1,3,imsize,imsize)
    image = image.astype(np.float32)
    return image

""" To save non-preprocessed images as jpeg:
misc.imsave('frame2.jpeg', images[2,:,:,:])
misc.imsave('frame5.jpeg', images[5,:,:,:])
"""
def sendCommand( command ):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12345
        s.connect((host,port))
        s.send(command.encode()) 
        s.close()
    except Exception as inst:
        return False

    return True

class Drive:
    def __init__(self):
        self.stopped = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # I'm not 100% sure this will result in everything being closed. Best to also call stop().
        self.stopped = True

    def stop(self,signum=0,frame=0):
        # indicate that the thread should be stopped
        self.stopped = True

    def startDriving( self, options ):
        if not os.path.exists(options.dir_ep):
            os.makedirs(options.dir_ep)
        video_path = os.path.join( config.directories['video'], options.episode+".h264" )

        images = []
        actions = []
        action_times = []

        framerate = 32
        brightness = 60
        imsize = 120
        log( 'Camera framerate: ' + str(framerate), options )
        log( 'Camera brightness: ' + str(brightness), options )

        actions_path = os.path.abspath( os.path.join( options.dir_ep, options.episode+"_actions.text" ) )

        accel_path = os.path.join( options.dir_ep, options.episode+".accel" )
        accel = Popen(['./accelerometer/accelerometer.py', accel_path])

        with PiVideoStream( resolution=(480,480), imsize=imsize, framerate=framerate, brightness=brightness ) as vs:
            vs.start()
            sleep(1) # Let the camera warm up

            log( "Start drive", options )

            if options.video:
                vs.startVideo(video_path)

            sendCommand( 'record ' + actions_path )

            print( "Start driving!!!" )

            t_start = time()
            time_steps = options.steps
            for x in range(time_steps):
                (full,image) = vs.read()
                if full is not None:
                    images.append( full )
                sleep(0.1)

                if self.stopped:
                    break
            vs.stop()

        accel.terminate()
        sendCommand( 'record' )

        log( "Stop drive", options )
        print "Drive elapsed time: %f" % ((time() - t_start),)
        sleep(2)

# Move the video file to the episode directory
        if options.video:
            if os.path.exists(video_path):
                shutil.move( video_path, os.path.join(options.dir_ep, options.episode+".h264") )
            else:
                print "Video file is missing: %s" % (video_path,)

        accel_data = []
        with open(accel_path, 'rb') as f:
            accel_data = pickle.load( f )

        print "Writing Episode Data"    
        pkg = self.packageImages( images, actions, action_times, accel_data, options )
        images_filename = os.path.join( options.dir_ep, options.episode + "_episode.pickle" )
        with open(images_filename, 'wb') as f:
            pickle.dump( pkg, f, pickle.HIGHEST_PROTOCOL)
        print "Finished Writing Episode Data"    

    def packageImages(self, images, actions, action_times, accel_data, options ):
        image_pkg = { }
        image_pkg["episode"] = options.episode
        image_pkg["format"] = options.im_format
        image_pkg["date"] = datetime.datetime.now()
        image_pkg["model"] = "Manual Drive"
        image_pkg["actions"] = actions
        image_pkg["action_times"] = action_times
        image_pkg["accelerometer"] = accel_data
        if options.im_format == "numpy":
            image_pkg["images"] = images
        elif options.im_format == "jpeg":
            jpegs = []
            for image in images:
                output = cStringIO.StringIO()
                jpeg = misc.imsave(output, image, format='jpeg')
                jpegs.append(jpeg)
            image_pkg["images"] = jpegs
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

    with Drive() as adrive:
        adrive.startDriving( options )
