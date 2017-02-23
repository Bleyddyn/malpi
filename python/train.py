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

from malpi.model import *
from malpi.solver import *


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

def loadEpisode( options, verbose=True ):
    """ Will return a dictionary of episode data from the pickle file, or None if the load failed.  """
    filename = os.path.join( options.dir_ep, options.episode + "_episode.pickle" )
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

def crop_center( img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocessOneImage(image, imsize):
#image.shape (480, 720, 3)
    image = image.transpose(2,1,0)
    width = image.shape[1]
    # shape = (3, 720, 480)
    if width > 480:
        min = (width - 480) / 2
        image = image[:,min:min+480,:]
    image = misc.imresize(image,(imsize,imsize))
    # shape = (3, imsize, imsize)
    image = image.reshape(1,3,imsize,imsize)
    image = image.astype(np.float32)
    return image

""" To save non-preprocessed images as jpeg:
misc.imsave('frame2.jpeg', images[2,:,:,:])
misc.imsave('frame5.jpeg', images[5,:,:,:])
"""

def actionToCommand(action):
    commands = ["forward","backward","left","right","stop"]
    return commands[action]

def softmax(x):
  probs = np.exp(x - np.max(x))
  probs /= np.sum(probs )
  return probs

def trainEpisode( options ):
    model = loadModel( options, verbose=True )
    if not model:
        print "Error reading model: %s" % options.model_name
        return

    episode = loadEpisode( options, verbose=True )
    if not episode:
        print "Error reading episode: %s" % options.episode
        return

    solver = Solver(model)

    images = episode["images"]
    actions = episode["actions"]
    rewards = episode["rewards"]

def packageImages( images, actions, options ):
    image_pkg = { }
    image_pkg["episode"] = options.episode
    image_pkg["format"] = options.im_format
    image_pkg["date"] = datetime.datetime.now()
    image_pkg["model"] = options.model_name
    image_pkg["actions"] = actions
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

def writeJpegs( options ):
    episode = loadEpisode( options, verbose=True )
    if not episode:
        print "Error reading episode: %s" % options.episode
        return

    images = episode["images"]

    dir_name = os.path.join( options.dir_ep, "jpegs" )

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pad = len(str(len(images)))
    for idx, image in enumerate(images):
        action =  actionToCommand( episode['actions'][idx] )
        fname = os.path.join( dir_name, str(idx).zfill(pad) + "_" + action + ".jpeg" )
        misc.imsave(fname, image, format='jpeg')

def addRewards( options ):
    episode = loadEpisode( options, verbose=True )
    if not episode:
        print "Error reading episode: %s" % options.episode
        return

    rewards = np.zeros(len(episode['images']))
    with open(options.reward) as f:
        for idx, line in enumerate(f):
            rewards[idx] = float(line)

    episode['rewards'] = rewards

    filename = os.path.join( options.dir_ep, options.episode + "_episode.pickle" )
    with open(filename, 'wb') as f:
        pickle.dump( episode, f, pickle.HIGHEST_PROTOCOL)

def describe( options ):
    episode = loadEpisode( options, verbose=True )
    if not episode:
        print "Error reading episode: %s" % options.episode
        return

    model = loadModel( options, verbose=True )
    if not model:
        print "Error reading model: %s" % options.model_name
        return

    model.describe()

    print

    for key, val in episode.iteritems():
        if key == 'images' or key == 'actions' or key == 'rewards':
            print "%s: %d" % (key, len(val))
        else:
            print "%s: %s" % (key, val)

def getOptions():
    usage = "Usage: python ./train.py <model name> <episode name> [<episode2 name>...]"
    parser = OptionParser()
    parser.add_option("-l","--list", help="Text file with a list of episodes, one per line.");
    parser.add_option("-i","--initialize", action="store_true", default=False, help="Initialize cnn and lstm models, save them to <model name>.pickle, then exit.");
    parser.add_option("--dir_ep", help="Directory for saving all episode related files. Defaults to episode name.");
    parser.add_option("-d","--dir_model", default="", help="Directory for finding/initializing model files. Defaults to current directory.");
    parser.add_option("-j","--jpeg", action="store_true", default=False, help="Convert all images to jpegs and write to <dir_ep>/jpegs/.");
    parser.add_option("-r","--reward", default=None, help="Text file with rewards to be added to the episode. One per line. Count must match number of images/actions in episode.");
    parser.add_option("--test_only", action="store_true", default=False, help="Run tests, then exit.");
    parser.add_option("--desc", action="store_true", default=False, help="Describe the model and episode, then exit.");

    (options, args) = parser.parse_args()

    if len(args) != 2:
        print usage
        exit()

    options.dir_ep = os.path.expanduser(options.dir_ep)
    options.dir_model = os.path.expanduser(options.dir_model)

    if args[0].endswith('.pickle'):
        args[0] = args[0][:-7]

    options.model_name = args[0]

    if args[1].endswith('.pickle'):
        args[1] = args[1][:-7]

    options.episode = args[1]
    if not options.dir_ep:
        options.dir_ep = options.episode
    else:
        options.dir_ep = os.path.join( options.dir_ep, options.episode )

    # Hard code this for now
    options.im_format = "numpy"

    return (options, args)

def test(options):
    print options.dir_ep
    print options.dir_model
    #testImages( options )

if __name__ == "__main__":
    (options, args) = getOptions()

    if options.test_only:
        test(options)
        exit()

    if options.jpeg:
        writeJpegs( options )
        exit()

    if options.initialize:
        print "Initializing cnn and lstm models..."
        initializeModel( options )
        exit()

    if options.reward:
        addRewards( options )
        exit()

    if options.desc:
        describe( options )
        exit()

    try:
        runEpisode( options )
    except (Exception, KeyboardInterrupt, SystemExit):
        print "Stopping motors"
        sendMotorCommand( "stop", options ) # always stop the motors before ending
        raise
