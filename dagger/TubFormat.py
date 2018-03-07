import os
import pickle
from DriveFormat import DriveFormat
from collections import defaultdict
import numpy as np
import json
import re
from scipy.ndimage import imread

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
class TubFormat(DriveFormat):
    """ A class to represent a DonkeyCar Tub drive on disc.
    """

    def __init__( self, path ):
        DriveFormat.__init__(self)

        if not os.path.exists(path):
            raise IOError( "TubFormat directory does not exist: {}".format( path ) )
        if not os.path.isdir(path):
            raise IOError( "TubFormat path is not a directory: {}".format( path ) )

        self.path = path
        (self.images, self.actions) = self._load(path)
        self.meta = self._loadMeta(path)

    def _loadMeta( self, path ):
        """ e.g. {"inputs": ["cam/image_array", "user/angle", "user/throttle", "user/mode"], "types": ["image_array", "float", "float", "str"]}
        """

        meta_file = os.path.join( path, "meta.json" )
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file,'r') as f:
                meta = json.load(f)
        return str(meta)

    def _load( self, path, image_norm=True ):
        images = []
        actions = []

        image_files = []
        action_files = []
        for fname in os.listdir(path):
            if "cam-image_array_" in fname and fname.endswith(".jpg"):
                image_files.append(fname)
            elif fname.startswith("record_") and fname.endswith(".json"):
                action_files.append(fname)

        image_files.sort(key=natural_keys)
        action_files.sort(key=natural_keys)

        for i in range(len(image_files)):
            images.append( imread( os.path.join( self.path, image_files[i] ) ) )
            with open( os.path.join( self.path, action_files[i] ) ) as f:
                actions.append( json.load(f)["user/angle"] )

        return images, actions

    def save( self ):
        # Ignoring images for now
        # For tub files it might make sense to keep a list of modified actions so only those need to be written out
        self.setClean()

    def count( self ):
        return len(self.images)

    def imageForIndex( self, index ):
        return self.images[index]

    def actionForIndex( self, index ):
        return str(self.actions[index])

    def setActionForIndex( self, new_action, index ):
        if self.actions[index] != new_action:
            self.actions[index] = new_action
            self.setDirty()

    def actionNames(self):
        return [ ]

    def actionForKey(self,keybind,oldAction=None):
        if keybind == 'w':
            return 'forward'
        elif keybind == 'a':
            return 'left'
        elif keybind == 'd':
            return 'right'
        elif keybind == 's':
            return 'stop'
        elif keybind == 'x':
            return 'backward'
        return None

    def actionStats(self):
        stats = defaultdict(int)
        for action in self.actions:
            stats[action] += 1
        return stats

    @classmethod
    def canOpenFile( cls, path ):
        if not os.path.exists(path):
            return False
        if not os.path.isdir(path):
            return False

        meta_file = os.path.join( path, "meta.json" )
        if not os.path.exists(meta_file):
            return False

        #if os.path.splitext(path)[1] == ".tub":
        #    return True

        return True

""" Register this format with the base class """
DriveFormat.registerFormat( "TubFormat", TubFormat )

if __name__ == "__main__":
    DriveFormat.testFormat( TubFormat, "test.tub", 2.0 )
