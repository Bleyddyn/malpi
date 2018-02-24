import os
import numpy as np
import pickle

class Drive:
    """ A class to represent a MaLPi drive on disc.
    """

    def __init__( self, path ):
        if not os.path.exists(path):
            raise IOError( "Drive directory does not exist: {}".format( path ) )
        if not os.path.isdir(path):
            raise IOError( "Drive path is not a directory: {}".format( path ) )

        self.path = path
        (self.images, self.actions) = self._load(path)
        self.meta = self._loadMeta(path)
        self.clean = True

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
            actions = actions.astype('str')
        else:
            actions_file = os.path.join( drive_dir, "image_actions.pickle" )
            with open(actions_file,'r') as f:
                actions = pickle.load(f)

        im_file = os.path.join( drive_dir, "images_120x120.npy" )
        if os.path.exists(im_file):
            images = np.load(im_file)
        else:
            im_file = os.path.join( drive_dir, "images_120x120.pickle" )
            with open(im_file,'r') as f:
                images = pickle.load(f)

        return images, actions

    def _load( self, path, image_norm=True ):
        images = []
        actions = []

        images, actions = self._loadOneDrive( path )

        images = np.array(images)

        return images, actions

    def save( self ):
        # Ignoring images for now
        actions = np.array(self.actions)
        actions = actions.astype('str')
        ofname = os.path.join( self.path, 'image_actions.npy' )
        np.save(ofname, actions)

    def isClean( self ):
        return self.clean

    def imageForIndex( self, index ):
        return self.images[index]

    def actionForIndex( self, index ):
        return self.actions[index]

    def setActionForIndex( self, action, index ):
        if self.actions[index] != action:
            self.actions[index] = action
            self.clean = False

    @staticmethod
    def actionNames():
        return [ "forward", "backward", "left", "right", "stop" ]

def tests():

    test_path = "test.drive"

    d = Drive(test_path)

    print( "Drive:\n{}".format( d.meta ) )
    print( "Actions: {}".format( Drive.actionNames() ) )
    print( "Image 10 shape: {}".format( d.imageForIndex(9).shape ) )
    print( "Action 10: {}".format( d.actionForIndex(9) ) )
    
    try:
        d = Drive("DriveFormat.py")
    except IOError as ex:
        print( "Caught correct exception when path is not a directory: PASS" )
    except Exception as exg:
        print( "Caught invalid exception ({}) when path is not a directory: FAIL".format(exg) )
    else:
        print( "No exception raised when path is not a directory: FAIL" )

    try:
        d = Drive("NonExistantDrivePath_________")
    except IOError as ex:
        print( "Caught correct exception when path does not exist: PASS" )
    except Exception as exg:
        print( "Caught invalid exception ({}) when path does not exist: FAIL".format(exg) )
    else:
        print( "No exception raised when path does not exist: FAIL" )

if __name__ == "__main__":
    tests()
