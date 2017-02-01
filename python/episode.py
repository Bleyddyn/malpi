import os
from time import time
import datetime
from optparse import OptionParser
import socket
import cStringIO
import io

import numpy as np
from scipy import ndimage
from scipy import misc

from malpi.cnn import *
from malpi.lstm import *
from malpi.data_utils import get_CIFAR10_data
from malpi.solver import Solver
from malpi.rnn_layers import *

from picamera import PiCamera

def initializeModels( name ):
    imsize = 79
    layers = ["conv-8", "maxpool", "conv-16", "maxpool", "conv-32"]
    layer_params = [{'filter_size':3, 'stride':2, 'pad':1 }, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3}, {'pool_stride':2, 'pool_width':2, 'pool_height':2},
        {'filter_size':3} ]
    model = MalpiConvNet(layers, layer_params, input_dim=(3,imsize,imsize), reg=.005, dtype=np.float32, verbose=True)
    model.name = name
    model.describe()

    N = 1
    D = 32*10*10
    H = 500
    nA = 5

    lstm_model = MalpiLSTM( D, H, nA, dtype=np.float32 )
    lstm_model.name = name
    lstm_model.describe()

    filename = name + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump( (model,lstm_model), f, pickle.HIGHEST_PROTOCOL)

def loadModels( model_name, verbose=True ):
    """ Will return a tuple of (cnn_model, lstm_model), or (None,None) if the load failed.  """
    filename = model_name+".pickle"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IOError as err:
        if verbose:
            print("IO error: {0}".format(err))
        return (None,None)

def log( message, name, directory ):
    logFileName = os.path.join(directory, name) + ".log"
    fmt = '%Y-%m-%d-%H-%M-%S'
    datestr = datetime.datetime.now().strftime(fmt)

    with open(logFileName,'a') as outf:
        outf.write(datestr + ": " + message + "\n")

def getOneTestImage(imsize):
    # Get one image (480x480) from the Camera Daemon
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
# input_dim: Tuple (C, H, W) giving size of input data.

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
# input_dim: Tuple (C, H, W) giving size of input data.

def readOneJPEG():
    try:
        command = 'image'
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12346
        s.connect((host,port))
        s.sendall(command.encode()) 
        s.shutdown(socket.SHUT_WR)
        sfile = s.makefile('r')
        jpeg_byte_string = sfile.read() 
        strobj = cStringIO.StringIO(jpeg_byte_string)
        image = ndimage.imread(strobj)
        sfile.close()
        s.close()
        return image
    except Exception as inst:
        print "Error in readOneJPEG: %s" % str(inst)
        return None

    return None

def readOneJPEG_local():
    camera = PiCamera()
    my_stream = io.BytesIO()
    camera.capture(my_stream, 'jpeg')
    my_stream.seek(0)
    image = ndimage.imread(my_stream)
    return image

def testOneImage():
    jpeg = readOneJPEG()

def softmax(x):
  probs = np.exp(x - np.max(x))
  probs /= np.sum(probs )
  return probs

def runEpisode( model_name, episode_name ):
    (model, lstm_model) = loadModels( model_name, verbose=True )
    if not model or not lstm_model:
        print "Error reading model: %s" % model_name
        return

    if not os.path.exists(episode_name):
        os.makedirs(episode_name)

    # Tell the Camera Daemon to start recording video
    log( "Start episode", model_name, episode_name )
    imsize = model.input_dim[1]

    t_start = time()
    time_steps = 10
    for x in range(time_steps):
        t1 = time()
        image = readOneJPEG()
        t2 = time()
        image = preprocessOneImage(image, imsize)
        t3 = time()
        cnn_out = model.loss(image)
        actions = lstm_model.loss(cnn_out)
        t4 = time()
        actions = actions[0]
        # Sample an action
        action = np.random.choice(np.arange(len(actions)), p=actions)
        log( "Action: " + str(action), model_name, episode_name )
        print "%f - %f - %f" % ( (t2 - t1), (t3 - t2), (t4 - t3))
# 0.872346 - 0.133617 - 0.186266

    log( "Stop episode", model_name, episode_name )
    # Tell the Camera Daemon to stop recording video
# Move the video file to the episode directory
    
    print "Episode elapsed time: %f" % ((time() - t_start),)

def getOptions():
    usage = "Usage: python ./episode.py [--name=<ep name>] <model name>"
    parser = OptionParser()
    parser.add_option("-n","--name",help="Session Name. Used for directory and log file names. Defaults to date/time.");
    parser.add_option("-i","--initialize",action="store_true", default=False,help="Initialize cnn and lstm models, save them to <model name>.pickle, then exit.");
    (options, args) = parser.parse_args()
    if not options.name:
        n = datetime.datetime.now()
        options.name = n.strftime('%Y%m%d_%H%M%S') 
    if len(args) == 0:
        print usage
        exit()
    return (options, args)

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.initialize:
        print "options.init"
        initializeModels( args[0] )
        exit()

    runEpisode( args[0], options.name )
