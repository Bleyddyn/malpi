import os
from time import time
from time import sleep
import datetime
from optparse import OptionParser
import socket
import cStringIO
import io
import shutil

import numpy as np
from scipy import ndimage
from scipy import misc

from malpi.cnn import *
from malpi.lstm import *
from malpi.data_utils import get_CIFAR10_data
from malpi.solver import Solver
from malpi.rnn_layers import *

from picamera import PiCamera

try:
    import config
except:
    print "Failed to load config file config.py."
    print "Try copying config_empty.py to config.py and re-running."
    exit()

def initializeModels( options ):
    imsize = 79
    layers = ["conv-8", "maxpool", "conv-16", "maxpool", "conv-32"]
    layer_params = [{'filter_size':3, 'stride':2, 'pad':1 }, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3} ]
    model = MalpiConvNet(layers, layer_params, input_dim=(3,imsize,imsize), reg=.005, dtype=np.float32, verbose=True)
    model.name = options.model_name

    print
    model.describe()
    print

    N = 1
    D = 32*10*10
    H = 500
    nA = 5

    lstm_model = MalpiLSTM( D, H, nA, dtype=np.float32 )
    lstm_model.name = options.model_name

    lstm_model.params['bo'][0] += 1 # Bias the output to move the robot forward

    lstm_model.describe()

    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( (model,lstm_model), f, pickle.HIGHEST_PROTOCOL)

def loadModels( options, verbose=True ):
    """ Will return a tuple of (cnn_model, lstm_model), or (None,None) if the load failed.  """
    filename = os.path.join( options.dir_model, options.model_name + ".pickle" )
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IOError as err:
        if verbose:
            print("IO error: {0}".format(err))
        return (None,None)

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

def preprocessOneImage(image, imsize):
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

def sendCameraCommand(command):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12346
        s.connect((host,port))
        s.sendall(command.encode()) 
        s.shutdown(socket.SHUT_WR)
        if command == 'image':
            sfile = s.makefile('r')
            jpeg_byte_string = sfile.read() 
            strobj = cStringIO.StringIO(jpeg_byte_string)
            image = ndimage.imread(strobj)
            sfile.close()
        else:
            image = None
        s.close()
        return image
    except Exception as inst:
        print "Error in sendCameraCommand: %s" % str(inst)
        return None

    return None

def readOneJPEG_local():
    camera = PiCamera()
    my_stream = io.BytesIO()
    camera.capture(my_stream, 'jpeg')
    my_stream.seek(0)
    image = ndimage.imread(my_stream)
    return image

def softmax(x):
  probs = np.exp(x - np.max(x))
  probs /= np.sum(probs )
  return probs

def runEpisode( options ):
    (model, lstm_model) = loadModels( options, verbose=True )
    if not model or not lstm_model:
        print "Error reading model: %s" % options.model_name
        return

    if not os.path.exists(options.dir_ep):
        os.makedirs(options.dir_ep)

    sendCameraCommand('video_start '+options.episode) # Tell the Camera Daemon to start recording video
    sleep(1) # seems to be necessary
    log( "Start episode", options )
    imsize = model.input_dim[1]
    test_image = getOneTestImage(imsize)

    t_start = time()
    time_steps = 10
    for x in range(time_steps):
        #image = sendCameraCommand('image')
        #image = preprocessOneImage(image, imsize)
        image = test_image
        cnn_out = model.loss(image)
        actions = lstm_model.loss(cnn_out)
        actions = actions[0]
        # Sample an action
        action = np.random.choice(np.arange(len(actions)), p=actions)
        log( "Action: " + str(action), options )
        #print "%f - %f - %f" % ( (t2 - t1), (t3 - t2), (t4 - t3))
# 0.872346 - 0.133617 - 0.186266

    log( "Stop episode", options )
    sendCameraCommand('video_stop') # Tell the Camera Daemon to stop recording video
    sleep(1)

# Move the video file to the episode directory
    video_path = os.path.join(config.directories['video'],options.episode+".h264")
    if os.path.exists(video_path):
        shutil.move( video_path, os.path.join(options.dir_ep, options.episode+".h264") )
    else:
        print "Video file is missing: %s" % (video_path,)
    
    print "Episode elapsed time: %f" % ((time() - t_start),)

def getOptions():
    usage = "Usage: python ./episode.py [--name=<ep name>] <model name>"
    parser = OptionParser()
    parser.add_option("-e","--episode", help="Episode Name. Used for episode related file names. Defaults to date/time.");
    parser.add_option("-i","--initialize",action="store_true", default=False, help="Initialize cnn and lstm models, save them to <model name>.pickle, then exit.");
    parser.add_option("--dir_ep", help="Directory for saving all episode related files. Defaults to episode name.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        exit()

    if not options.episode:
        n = datetime.datetime.now()
        options.episode = n.strftime('%Y%m%d_%H%M%S') 

    if not options.dir_ep:
        options.dir_ep = options.episode

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]

    options.model_name = args[0]

    return (options, args)

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.initialize:
        print "Initializing cnn and lstm models..."
        initializeModels( options )
        exit()

    runEpisode( options )
