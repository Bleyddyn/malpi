import os
import pickle
import json
from malpi.ui.DriveFormat import DriveFormat
from collections import defaultdict
import numpy as np

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

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
        self.auxMeta = {}
        self.auxData = {}
        self.meta = self._loadMeta(path)
        #self._load(path)

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
            actions = np.array(actions,dtype=str)
        else:
            actions_file = os.path.join( drive_dir, "image_actions.pickle" )
            with open(actions_file,'r') as f:
                actions = pickle.load(f)
                actions = np.array(actions,dtype=str)

        categorical = True
        if isinstance(actions[0], basestring):
            actions = actions.astype(str)
            act_str = []
            for act in actions:
                act_str.append(str(act))
            actions = act_str
            #actions = embedActions( actions )
            #actions = to_categorical( actions, num_classes=5 )
            categorical = True
        elif type(actions) == list:
            actions = np.array(actions).astype('float')
            act_str = []
            for act in actions:
                act_str.append(act)
            actions = act_str
            categorical = False
        elif type(actions) == np.ndarray:
            actions = np.array(actions).astype('float')
            act_str = []
            for act in actions:
                act_str.append(act)
            actions = act_str
            categorical = False
        else:
            print("Unknown actions format: {} {} as {}".format( type(actions), actions[0], type(actions[0]) ))

        self.categorical = categorical

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

        # self.images needs to be set before loading aux data, in case we need to fill in default values
        self.images = images
        self.actions = actions

        aux_file = os.path.join( self.path, "aux.json" )
        if os.path.exists(aux_file):
            with open(aux_file,'r') as f:
                self.auxMeta = json.load(f)

            for key, value in self.auxMeta.items():
                ofname = os.path.join( self.path, key+'_aux.npy' )
                if not os.path.exists(ofname):
                    print( "Adding missing auxiliary data file: {}".format( ofname ) )
                    self.addAuxData(value)
                else:
                    data = np.load(ofname)
                    data_str = []
                    for act in data:
                        data_str.append(str(act))
                    self.auxData[key] = data_str

        return images, actions

    def load( self, progress=None ):
        (self.images, self.actions) = self._load(self.path)
        self.setClean()

    def save( self ):
        # Ignoring images for now
        images = np.array(self.images)
        ofname = os.path.join( self.path, 'images_120x120.npy' )
        np.save(ofname, images)

        actions = np.array(self.actions)
        actions = actions.astype('str')
        ofname = os.path.join( self.path, 'image_actions.npy' )
        np.save(ofname, actions)

        aux_file = os.path.join( self.path, "aux.json" )
        with open(aux_file,'w') as f:
            json.dump(self.auxMeta,f)

        for key, value in self.auxData.items():
            data = np.array(value)
            ofname = os.path.join( self.path, key+'_aux.npy' )
            np.save(ofname, data)

        self.setClean()


    def count( self ):
        return len(self.images)

    def imageForIndex( self, index ):
        return self.images[index]

    def actionForIndex( self, index ):
        return self.actions[index]

    def setActionForIndex( self, new_action, index ):
        diff = False
        if self.categorical:
            if self.actions[index] != new_action:
                diff = True
        else:
            if not np.array_equal( self.actions[index], new_action ):
                diff = True

        if diff:
            self.actions[index] = new_action
            self.setDirty()

    def deleteIndex( self, index ):
        if index >= 0 and index < self.count():
            self.images = np.delete(self.images,index,axis=0)
            self.actions.pop(index)
            for key in self.auxData:
                self.auxData[key].pop(index)
            self.setDirty()

    def actionForKey(self,keybind,oldAction=None):
        if self.categorical:
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
        else:
            oldAction = np.copy(oldAction)
            if keybind == 'w':
                oldAction[1] += 0.1
            elif keybind == 'a':
                oldAction[0] -= 0.1
            elif keybind == 'd':
                oldAction[0] += 0.1
            elif keybind == 's':
                oldAction = [0.0,0.0]
            elif keybind == 'x':
                oldAction[1] -= 0.1
            return np.clip(oldAction, -1.0, 1.0)
        return None

    def metaString(self):
        return self.meta

    def actionStats(self):
        if self.categorical:
            stats = defaultdict(int)
            for action in self.actions:
                stats[action] += 1
            return stats
        else:
            stats = {}
            stats["Mean"] = np.mean(self.actions)
            stats["StdDev"] = np.std(self.actions)
            return stats

    def supportsAuxData(self):
        return True

    def getAuxMeta(self):
        return self.auxMeta

    def addAuxData(self, meta):
        # TODO Check to make sure the meta data is all the same
        if meta["name"] not in self.auxMeta:
            self.auxMeta[meta["name"]] = meta
        if meta["name"] not in self.auxData:
            self.auxData[meta["name"]] = [meta["default"]]*self.count()

    def auxDataAtIndex(self, auxName, index):
        if not auxName in self.auxData:
            return None
        return self.auxData[auxName][index]

    def setAuxDataAtIndex(self, auxName, auxData, index):
        if not auxName in self.auxData:
            return False
        if self.auxData[auxName][index] != auxData:
            self.auxData[auxName][index] = auxData
            self.setDirty()
        return True

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
        if self.categorical:
            res = MalpiFormat.defaultOutputTypes()
        else:
            res = [{"name":"Actions", "type":"continuous" }]
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
