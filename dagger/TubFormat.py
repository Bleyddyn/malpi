import os
import sys
import pickle
from DriveFormat import DriveFormat
from collections import defaultdict
import numpy as np
import json
import re
from scipy.ndimage import imread
from donkeycar.parts.datastore import Tub

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
        self.tub = Tub(path)
        self.exclude = set()
        self.shape = None
        #(self.images, self.actions) = self._load(path)
        self._meta = self._loadMeta(path)
        self.meta = str(self._meta)

    def _loadMeta( self, path ):
        """ e.g. {"inputs": ["cam/image_array", "user/angle", "user/throttle", "user/mode"], "types": ["image_array", "float", "float", "str"]}
        """

        meta_file = os.path.join( path, "meta.json" )
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file,'r') as f:
                meta = json.load(f)

        exc_file = os.path.join( path, "exclude.json" )
        if os.path.exists(exc_file):
            with open(exc_file,'r') as f:
                excl = json.load(f) # stored as a list
                self.exclude = set(excl)

        return meta

    def _load( self, path, image_norm=True, progress=None ):
        images = []
        actions = []

        num_records = self.tub.get_num_records()
        for i in range(0,num_records+1):
            try:
                #print( "Loading {} of {}".format( i, num_records ), end='\r' )
                if progress is not None:
                    progress( i, num_records )
                sys.stdout.flush()
                rec = self.tub.get_record(i)
                if 'cam/image_array' in rec:
                    images.append( rec['cam/image_array'] )
                    if self.shape is None:
                        self.shape = rec['cam/image_array'].shape
                    one_action = []
                    for act in ["user/angle", "user/throttle"]:
                        if act in rec:
                            one_action.append( rec[act] )
                    actions.append( one_action )
                    #actions.append( rec["user/throttle"] )
            except Exception as ex:
                if i > 0 and i < num_records:
                    # Some tubs are zero indexed, others 1. So this is only sometimes a real error
                    print("Load failed: {}".format( ex ) )
        #print("")

        return images, actions

    def load( self, progress=None ):
        (self.images, self.actions) = self._load(self.path, progress=progress)
        self.setClean()

    def save( self ):
        if self.isClean():
            return

        exc_file = os.path.join( self.path, "exclude.json" )
        with open(exc_file,'w') as f:
            json.dump( list(self.exclude), f )
        
        # For tub files it might make sense to keep a list of modified actions so only those need to be written out
        self.setClean()

    def count( self ):
        return len(self.images)

    def imageForIndex( self, index ):
        if (index + 1) in self.exclude:
# This ends up looking ugly, can't figure out why
            tmp = self.images[index].mean(axis=-1,dtype=self.images[index].dtype,keepdims=False)
            tmp = np.repeat( tmp[:,:,np.newaxis], 3, axis=2 )
            return tmp
        return self.images[index]

    def actionForIndex( self, index ):
        return self.actions[index]

    def setActionForIndex( self, new_action, index ):
        if not np.array_equal( self.actions[index], new_action ):
            self.actions[index] = new_action
            self.setDirty()

    def actionForKey(self,keybind,oldAction=None):
        if keybind == 'w':
            oldAction[1] += 0.1
        elif keybind == 'x':
            oldAction[1] -= 0.1
        elif keybind == 'a':
            oldAction[0] -= 0.1
        elif keybind == 'd':
            oldAction[0] += 0.1
        elif keybind == 's':
            oldAction[0] = 0.0
            oldAction[1] = 0.0
        return np.clip(oldAction, -1.0, 1.0)

    def deleteIndex( self, index ):
        if index >= 0 and index < self.count():
            index += 1
            if index in self.exclude:
                self.exclude.remove(index)
            else:
                self.exclude.add(index)
            self.setDirty()

    def actionStats(self):
        stats = defaultdict(int)
        if len(self.actions) > 0:
            stats["Min"] = np.min(self.actions)
            stats["Max"] = np.max(self.actions)
            stats["Mean"] = np.mean(self.actions)
            stats["StdDev"] = np.std(self.actions)
        return stats

    def getAuxMeta(self):
        return {}

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

    @staticmethod
    def defaultInputTypes():
        return [{"name":"Images", "type":"numpy image", "shape":(120,160,3)}]

    def inputTypes(self):
        res = TubFormat.defaultInputTypes()
        if len(self.images) > 0:
            res[0]["shape"] = self.images[0].shape
        return res

    @staticmethod
    def defaultOutputTypes():
        return [{"name":"Actions", "type":"continuous", "range":(-1.0,1.0)}]

    def outputTypes(self):
        res = []
        for act in ["user/angle", "user/throttle"]:
            display_name = act.split("/")[1]
            res.append( {"name":display_name, "type":"continuous", "range":(-1.0,1.0)} )
        return res

""" Register this format with the base class """
DriveFormat.registerFormat( "TubFormat", TubFormat )

def runTests(args):
    DriveFormat.testFormat( TubFormat, args.file[0], 2.0 )
    d = TubFormat(args.file[0])
    print( d.inputTypes() )
    print( d.outputTypes() )

def getOptions():

    import argparse

    parser = argparse.ArgumentParser(description='Test Tub file format.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', nargs=1, metavar="File", default="test.tub", help='Recorded Tub data file to open')
    parser.add_argument('--test_only', action="store_true", default=True, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()
