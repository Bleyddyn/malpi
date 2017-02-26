import os
from time import time
from time import sleep
import datetime
from optparse import OptionParser
import cStringIO
import io
import shutil

import numpy as np
from scipy import ndimage
from scipy import misc

from malpi.model import *
from accelerometer import accelerometer
from PiVideoStream import PiVideoStream
from motors import Motors

try:
    import config
except:
    print "Failed to load config file config.py."
    print "Try copying config_empty.py to config.py and re-running."
    exit()

def initializeModel( options ):
    imsize = 79
    layers = ["conv-8", "maxpool", "conv-16", "maxpool", "conv-32", "lstml", "FC-5"]
    layer_params = [{'filter_size':3, 'stride':2, 'pad':1 }, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3}, {'hidden':500}, {} ]
    model = MalpiModel(layers, layer_params, input_dim=(3,imsize,imsize), reg=.005, dtype=np.float32, verbose=True)
    model.name = options.model_name

    print
    model.describe()
    print

    model.params['b7'][0] += 1 # Bias the output of the final layer to move the robot forward

    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( model, f, pickle.HIGHEST_PROTOCOL)

def loadModel( options, verbose=True ):
    """ Will return a model from the pickle file, or None if the load failed.  """
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IOError as err:
        if verbose:
            print("IO error: {0}".format(err))
        return None

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

commands = ["forward","backward","left","right","stop"]
def actionToCommand(action):
    return commands[action]

def softmax(x):
  probs = np.exp(x - np.max(x))
  probs /= np.sum(probs )
  return probs

def runEpisode( options ):
    model = loadModel( options, verbose=True )
    if not model:
        print "Error reading model: %s" % options.model_name
        return

    if not os.path.exists(options.dir_ep):
        os.makedirs(options.dir_ep)

    images = []
    actions = []
    action_times = []

    motorSpeed = 230
    framerate = 32
    brightness = 60
    imsize = model.input_dim[1]
    log( 'Motor Speed: ' + str(motorSpeed), options )
    log( 'Camera framerate: ' + str(framerate), options )
    log( 'Camera brightness: ' + str(brightness), options )

    accel = accelerometer.Accelerometer()

    with Motors() as motors:
        motors.setSpeed(motorSpeed)
        with PiVideoStream( resolution=(480,480), imsize=imsize, framerate=framerate, brightness=brightness ) as vs:
            vs.start()
            accel.start()
            sleep(1) # Let the camera warm up

            log( "Start episode", options )

            t_start = time()
            time_steps = options.steps
            for x in range(time_steps):
                (full,image) = vs.read()
                if full is not None:
                    images.append( full )
                    if image is None:
                        image = preprocessOneImage(full, imsize)
                    action_probs = model.loss(image)
                    action_probs = action_probs[0]
                    action_probs = softmax(action_probs)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # Sample an action
                    actions.append(action)
                    action_times.append(time())
                    motors.command( actionToCommand(action) )
                    print "Action %d: %s" % (x,actionToCommand(action))
                    #print "%f - %f - %f" % ( (t2 - t1), (t3 - t2), (t4 - t3))

    accel.stop()

    log( "Stop episode", options )
    print "Episode elapsed time: %f" % ((time() - t_start),)
    sleep(1)

# Move the video file to the episode directory
#    video_path = os.path.join(config.directories['video'],options.episode+".h264")
#    if os.path.exists(video_path):
#        shutil.move( video_path, os.path.join(options.dir_ep, options.episode+".h264") )
#    else:
#        print "Video file is missing: %s" % (video_path,)

    print "Writing Episode Data"    
    pkg = packageImages( images, actions, action_times, accel.read(), options )
    images_filename = os.path.join( options.dir_ep, options.episode + "_episode.pickle" )
    with open(images_filename, 'wb') as f:
        pickle.dump( pkg, f, pickle.HIGHEST_PROTOCOL)
    print "Finished Writing Episode Data"    

def packageImages( images, actions, action_times, accel_data, options ):
    image_pkg = { }
    image_pkg["episode"] = options.episode
    image_pkg["format"] = options.im_format
    image_pkg["date"] = datetime.datetime.now()
    image_pkg["model"] = options.model_name
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

def testImages( options ):
    with PiVideoStream( resolution=(480,480), imsize=79 ) as vs:
        vs.start()
        sleep(1) # seems to be necessary

        images = []
        for x in range(options.steps):
            (image, pre) = vs.read()
            if image is not None:
                images.append( image )
                print image.shape
                print pre.shape

#    #vs.stop()
#    print "Collected images"
#
#    if not os.path.exists(options.dir_ep):
#        os.makedirs(options.dir_ep)
#
#    options.im_format = "numpy"
#    t_start = time()
#    pkg = packageImages( images, [], options )
#    t1 = time()
#    print "Writing images"    
#    images_filename = os.path.join( options.dir_ep, options.episode + "_episode.pickle" )
#    with open(images_filename, 'wb') as f:
#        pickle.dump( pkg, f, pickle.HIGHEST_PROTOCOL)
#    t2 = time()
#    print "Finished Writing %d images" % (len(images),)
#    print "prepare: %f\n  write: %f" % (t1 - t_start, t2 - t1)

def getOptions():
    usage = "Usage: python ./episode.py [--name=<ep name>] <model name>"
    parser = OptionParser()
    parser.add_option("-e","--episode", help="Episode Name. Used for episode related file names. Defaults to date/time.");
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize cnn and lstm models, save them to <model name>.pickle, then exit.");
    parser.add_option("--dir_ep", help="Directory for saving all episode related files. Defaults to episode name.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-s","--steps", type="int", default=30, help="Number of steps to run. Defaults to 30.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        exit()

    if not options.episode:
        n = datetime.datetime.now()
        options.episode = n.strftime('%Y%m%d_%H%M%S') 

    if not options.dir_ep:
        options.dir_ep = options.episode
    else:
        options.dir_ep = os.path.join( options.dir_ep, options.episode )

    options.dir_ep = os.path.expanduser(options.dir_ep)
    options.dir_model = os.path.expanduser(options.dir_model)

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]

    options.model_name = args[0]

    # Hard code this for now
    options.im_format = "numpy"

    return (options, args)

def test(options):
    #print options.dir_ep
    print options.dir_model
    testImages( options )
    accel = accelerometer.Accelerometer()
    with Motors() as motor:
        motor.stop()
    exit()

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.test_only:
        test(options)

    if options.initialize:
        print "Initializing cnn and lstm models..."
        initializeModel( options )
        exit()

    runEpisode( options )
