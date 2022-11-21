import os
import sys
import pickle
import copy
from malpi.ui.DriveFormat import DriveFormat
from collections import defaultdict
import numpy as np
from PIL import Image
import json
import re
from donkeycar.parts.tub_v2 import Tub
from donkeycar.parts.datastore_v2 import NEWLINE
from donkeycar.pipeline.types import TubRecord
from donkeycar.utils import load_image

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
class Tubv2Format(DriveFormat):
    """ A class to represent a DonkeyCar Tub v2 on disc.

        Current assumptions:
            Tub records are 1 indexed and sequential with no gaps.
            We only care about editing steering and throttle.
            Steering and throttle should be clipped to -1/1.
    """

    def __init__( self, path ):
        DriveFormat.__init__(self)

        if not os.path.exists(path):
            raise IOError( "Tubv2Format directory does not exist: {}".format( path ) )
        if not os.path.isdir(path):
            raise IOError( "Tubv2Format path is not a directory: {}".format( path ) )

        self.path = path
        self.tub = Tub(path, read_only=False)
        self.meta = self.tub.manifest.metadata # Bug. tub.metadata doesn't get updated with info from disc
        self.deleted_indexes = self.tub.manifest.deleted_indexes
        if len(self.deleted_indexes) > 0:
            print( f"Deleted: {self.deleted_indexes}" )
        self.edit_list = set()
        self.shape = None

    def _load( self, path, image_norm=True, progress=None ):

        records = {}
        indexes = []
        images = [] # Store images separately so we can easily write changed records back to the tub
        total = len(self.tub)
        for idx, rec in enumerate(self.tub):
            img_path = os.path.join( self.path, self.tub.images(), rec['cam/image_array'] )
            try:
                img = Image.open( img_path )
                img_arr = np.asarray(img)
                if self.shape is None:
                    self.shape = img_arr.shape
            except Exception as ex:
                print( f"Failed to load image: {img_path}" )
                print( f"   Exception: {ex}" )
            records[idx] = rec
            indexes.append(idx)
            images.append(img_arr)
            progress(idx, total)
        self.records = records
        self.indexes = indexes
        self.images = images

    def load( self, progress=None ):
        self._load(self.path, progress=progress)
        self.setClean()

    def update_line( self, line_num, new_rec ):
        contents = json.dumps(new_rec, allow_nan=False, sort_keys=True)
        if contents[-1] == NEWLINE:
            line = contents
        else:
            line = f'{contents}{NEWLINE}'
        self.tub.manifest.current_catalog.seekable.update_line(line_num+1, line)

    def save( self ):
        if self.isClean():
            return

        self.tub.manifest.deleted_indexes = self.deleted_indexes

        for ix in self.edit_list:
            rec = self.records[ix]
            print( f"EditRecord: {rec}" )
            self.update_line( ix, rec )

        self.tub.manifest._update_catalog_metadata(update=True)
        self.edit_list.clear()
        self.setClean()

    def count( self ):
        return len(self.records)

    def imageForIndex( self, index ):
        idx = self.indexes[index]
        img = self.images[idx]
        if self.isIndexDeleted(index):
# This grayed out image ends up looking ugly, can't figure out why
            tmp = img.mean(axis=-1,dtype=img.dtype,keepdims=False)
            tmp = np.repeat( tmp[:,:,np.newaxis], 3, axis=2 )
            return tmp
        return img

    def get_angle_throttle(self, json_data):
        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        # If non-valid user entries and we have pilot data (e.g. AI), use that instead.
        if (0.0 == angle) and (0.0 == throttle):
            if "pilot/angle" in json_data:
                pa = json_data['pilot/angle']
                if pa is not None:
                    angle = float(pa)
            if "pilot/throttle" in json_data:
                pt = json_data['pilot/throttle']
                if pt is not None:
                    throttle = float(pt)

        return angle, throttle

    def actionForIndex( self, index ):
        idx = self.indexes[index]
        rec = self.records[idx]
        angle, throttle = self.get_angle_throttle(rec)
        return [angle, throttle]

    def setActionForIndex( self, new_action, index ):
        idx = self.indexes[index]
        rec = self.records[idx]
        angle, throttle = self.get_angle_throttle(rec)
        old_action = [angle, throttle]
        if not np.array_equal( old_action, new_action ):
            print( f"setActionForIndex" )
            if (rec["user/angle"] != new_action[0]) or (rec["user/throttle"] != new_action[1]):
                print( f"user: {rec['user/angle']}/{rec['user/throttle']}" )
                print( f"new: {new_action}" )
                # Save the original values if not already done
                if "orig/angle" not in rec:
                    rec["orig/angle"] = rec["user/angle"]
                if "orig/throttle" not in rec:
                    rec["orig/throttle"] = rec["user/throttle"]

                rec["user/angle"] = new_action[0]
                rec["user/throttle"] = new_action[1]
                self.edit_list.add(idx)
                self.setDirty()

    def actionForKey(self,keybind,oldAction=None):
        oldAction = copy.copy(oldAction)
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
        else:
            return None
        return np.clip(oldAction, -1.0, 1.0)

    def deleteIndex( self, index ):
        if index >= 0 and index < self.count():
            index += 1
            if index in self.deleted_indexes:
                self.deleted_indexes.remove(index)
            else:
                self.deleted_indexes.add(index)
            self.setDirty()

    def isIndexDeleted(self, index):
        if index >= 0 and index < self.count():
            index += 1
            return index in self.deleted_indexes
        return False

    def metaString(self):
        #{"inputs": ["cam/image_array", "user/angle", "user/throttle", "user/mode"], "start": 1550950724.8622544, "types": ["image_array", "float", "float", "str"]}
        ret = ""
        for k, v in self.meta.items():
            ret += "{}: {}\n".format( k, v )
        return ret

    def actionStats(self):
        stats = defaultdict(int)
        if self.count() > 0:
            actions = []
            for i in range(self.count()):
                act = self.actionForIndex( i )
                actions.append(act)
            stats["Min"] = np.min(actions)
            stats["Max"] = np.max(actions)
            stats["Mean"] = np.mean(actions)
            stats["StdDev"] = np.std(actions)
        return stats

    def supportsAuxData(self):
        return False

    def getAuxMeta(self):
        return None

    def addAuxData(self, meta):
        return None

    def auxDataAtIndex(self, auxName, index):
        return None

    def setAuxDataAtIndex(self, auxName, auxData, index):
        return False

    @classmethod
    def canOpenFile( cls, path ):
        if not os.path.exists(path):
            return False
        if not os.path.isdir(path):
            return False

        meta_file = os.path.join( path, "manifest.json" )
        if not os.path.exists(meta_file):
            return False

        return True

    @staticmethod
    def defaultInputTypes():
        return [{"name":"Images", "type":"numpy image", "shape":(120,160,3)}]

    def inputTypes(self):
        res = Tubv2Format.defaultInputTypes()
        if self.shape is not None:
            res[0]["shape"] = self.shape
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
DriveFormat.registerFormat( "Tubv2Format", Tubv2Format )

def runTests(args):
    DriveFormat.testFormat( Tubv2Format, args.file[0], [2.0, -5.5] )
    d = Tubv2Format(args.file[0])
    print( d.inputTypes() )
    print( d.outputTypes() )

def getOptions():

    import argparse

    parser = argparse.ArgumentParser(description='Test Tubv2 file format.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', nargs=1, metavar="File", default="test.tub", help='Recorded Tub data file to open')
    parser.add_argument('--test_only', action="store_true", default=True, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()
