import os
import sys
import pickle
import copy
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

        Current assumptions:
            Tub records are 1 indexed and sequential with no gaps.
            We only care about editing steering and throttle.
            Steering and throttle should be clipped to -1/1.

        TODO: Change actions to be a dictionary of dictionaries, with the outer key being a record's real index.
              Images would need to be included in that (which is how the Tub class does it).
    """

    def __init__( self, path ):
        DriveFormat.__init__(self)

        if not os.path.exists(path):
            raise IOError( "TubFormat directory does not exist: {}".format( path ) )
        if not os.path.isdir(path):
            raise IOError( "TubFormat path is not a directory: {}".format( path ) )

        self.path = path
        self.tub = Tub(path)
        self.meta = self.tub.meta
        self.edit_list = set()
        self.shape = None
        self.auxMeta = {}
        self.aux_clean = True
        #(self.images, self.actions) = self._load(path)

    def _load( self, path, image_norm=True, progress=None ):

        records = {}
        indexes = self.tub.get_index(shuffled=False)
        for idx in indexes:
            rec = self.tub.get_record(idx)
            if self.shape is None:
                self.shape = rec['cam/image_array'].shape
            records[idx] = rec
        self.records = records
        self.indexes = indexes

        if 'auxiliary' in self.tub.meta:
            self.auxMeta = self.tub.meta['auxiliary']

    def load( self, progress=None ):
        self._load(self.path, progress=progress)
        self.setClean()

    def save( self ):
        if not self.aux_clean:
            # update meta with new aux meta and write it out
            for name, aux in self.auxMeta.items():
                if name not in self.tub.meta['inputs']:
                    self.tub.meta['inputs'].append(name)
                    if 'continuous' == aux['type']:
                        aux_type = 'float'
                    elif 'categorical' == aux['type']:
                        aux_type = 'int'
                    else:
                        raise ValueError( "Unknown auxiliary data type: {}".format( aux['type'] ) )
                    self.tub.meta['types'].append(aux_type)
            self.tub.meta['auxiliary'] = self.auxMeta
            with open(self.tub.meta_path, 'w') as f:
                json.dump(self.tub.meta, f)
            self.aux_clean = True

        if self.isClean():
            return

        self.tub.write_exclude()

        for ix in self.edit_list:
            rec = self.records[ix]
            path = self.tub.get_json_record_path(ix)
            try:
                with open(path, 'r') as fp:
                    old_rec = json.load(fp)
            except TypeError:
                print('troubles with record:', path)
            except FileNotFoundError:
                raise
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise

            # Copy over only the keys we might have modified, this will need to include aux data at some point.
            for key in ['user/angle', 'user/throttle', 'orig/angle', 'orig/throttle']:
                if key in rec:
                    old_rec[key] = rec[key]

            try:
                with open(path, 'w') as fp: 
                    json.dump(old_rec, fp)
            except TypeError:
                print('troubles with record:', path)
            except FileNotFoundError:
                raise
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise

        self.edit_list.clear()
        self.setClean()

    def count( self ):
        return len(self.records)

    def imageForIndex( self, index ):
        idx = self.indexes[index]
        img = self.records[idx]['cam/image_array']
        if self.tub.excluded(index + 1):
# This grayed out image ends up looking ugly, can't figure out why
            tmp = img.mean(axis=-1,dtype=img.dtype,keepdims=False)
            tmp = np.repeat( tmp[:,:,np.newaxis], 3, axis=2 )
            return tmp
        return img

    def actionForIndex( self, index ):
        idx = self.indexes[index]
        rec = self.records[idx]
        angle, throttle = self.tub.get_angle_throttle(rec)
        return [angle, throttle]

    def setActionForIndex( self, new_action, index ):
        idx = self.indexes[index]
        rec = self.records[idx]
        angle, throttle = self.tub.get_angle_throttle(rec)
        old_action = [angle, throttle]
        if not np.array_equal( old_action, new_action ):
            if (rec["user/angle"] != new_action[0]) or (rec["user/throttle"] != new_action[1]):
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
        return np.clip(oldAction, -1.0, 1.0)

    def deleteIndex( self, index ):
        if index >= 0 and index < self.count():
            index += 1
            if self.tub.excluded(index):
                self.tub.include_index(index)
            else:
                self.tub.exclude_index(index)
            self.setDirty()

    def metaString(self):
        #{"inputs": ["cam/image_array", "user/angle", "user/throttle", "user/mode"], "start": 1550950724.8622544, "types": ["image_array", "float", "float", "str"]}
        ret = ""
        for k, v in self.tub.meta.items():
            ret += "{}: {}\n".format( k, v )
        return ret

    def actionStats(self):
        stats = defaultdict(int)
        if len(self.actions) > 0:
            stats["Min"] = np.min(self.actions)
            stats["Max"] = np.max(self.actions)
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
