import os
import pickle
from DriveFormat import DriveFormat
from collections import defaultdict
import numpy as np

class MalpiFormat(DriveFormat):
    """ A class to represent a MaLPi drive on disc.
    """

    def __init__( self, path ):
        DriveFormat.__init__(self)

        if not os.path.exists(path):
            raise IOError( "MalpiFormat directory does not exist: {}".format( path ) )
        if not os.path.isdir(path):
            raise IOError( "MalpiFormat path is not a directory: {}".format( path ) )

        self.path = path
        (self.images, self.actions) = self._load(path)
        self.meta = self._loadMeta(path)

    def _loadMeta( self, path ):
        meta_file = os.path.join( path, "meta.txt" )
        meta = ""
        if os.path.exists(meta_file):
            with open(meta_file,'r') as f:
                meta = f.read()
        return meta

    def _loadOneDrive( self, drive_dir ):
        actions_file = os.path.join( drive_dir, "image_actions.npy" )
        if os.path.exists(actions_file):
            actions = np.load(actions_file)
            actions = actions.astype(str)
        else:
            actions_file = os.path.join( drive_dir, "image_actions.pickle" )
            with open(actions_file,'r') as f:
                actions = pickle.load(f)
                actions = np.array(actions,dtype=str)

        im_file = os.path.join( drive_dir, "images_120x120.npy" )
        if os.path.exists(im_file):
            images = np.load(im_file)
        else:
            im_file = os.path.join( drive_dir, "images_120x120.pickle" )
            with open(im_file,'r') as f:
                images = pickle.load(f)
                images = np.array(images)

        return images, actions

    def _load( self, path, image_norm=True ):
        images = []
        actions = []

        images, actions = self._loadOneDrive( path )
        if len(images) != len(actions):
            print( "Images/actions: {}/{}".format( len(self.images), len(self.actions) ) )

        # A numpy array has a fixed string size, need this to be a python array
        act_str = []
        for act in actions:
            act_str.append(str(act))
        return images, act_str

    def save( self ):
        # Ignoring images for now
        actions = np.array(self.actions)
        actions = actions.astype('str')
        ofname = os.path.join( self.path, 'image_actions.npy' )
        np.save(ofname, actions)
        self.setClean()

    def count( self ):
        return len(self.images)

    def imageForIndex( self, index ):
        return self.images[index]

    def actionForIndex( self, index ):
        return self.actions[index]

    def setActionForIndex( self, new_action, index ):
        if self.actions[index] != new_action:
            self.actions[index] = new_action
            self.setDirty()

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

    @staticmethod
    def defaultInputTypes():
        return [{"name":"Images", "type":"numpy image", "shape":(120,120,3)}]

    def inputTypes(self):
        res = MalpiFormat.defaultInputTypes()
        if len(self.images) > 0:
            res[0]["shape"] = self.images[0].shape
        return res

    @staticmethod
    def defaultOutputTypes():
        return [{"name":"Actions", "type":"categorical", "categories":[ "forward", "backward", "left", "right", "stop" ]}]

    def outputTypes(self):
        res = MalpiFormat.defaultOutputTypes()
        return res

    @classmethod
    def canOpenFile( cls, path ):
        if not os.path.exists(path):
            return False
        if not os.path.isdir(path):
            return False

        have_action = False
        actions_file = os.path.join( path, "image_actions.npy" )
        if os.path.exists(actions_file):
            have_action = True
        else:
            actions_file = os.path.join( path, "image_actions.pickle" )
            if os.path.exists(actions_file):
                have_action = True

        have_images = False
        im_file = os.path.join( path, "images_120x120.npy" )
        if os.path.exists(im_file):
            have_images = True
        else:
            im_file = os.path.join( path, "images_120x120.pickle" )
            if os.path.exists(im_file):
                have_images = True

        if not have_action or not have_images:
            return False

        #if os.path.splitext(path)[1] == ".drive":
        #    return True

        return True

""" Register this format with the base class """
DriveFormat.registerFormat( "MalpiFormat", MalpiFormat )

if __name__ == "__main__":
    DriveFormat.testFormat( MalpiFormat, "test.drive", "very long action" )
    #d = MalpiFormat("test.drive")
    #print( d.inputTypes() )
    #print( d.outputTypes() )
