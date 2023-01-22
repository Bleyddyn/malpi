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

"""sqlite> .schema TubRecords
CREATE TABLE IF NOT EXISTS "TubRecords" (
    "source_id" INTEGER NOT NULL,
    "tub_index" INTEGER NOT NULL,
    "timestamp_ms" INTEGER,
    "image_path" TEXT,
    "mode" TEXT,
    "user_angle" REAL,
    "user_throttle" REAL,
    "pilot_angle" REAL,
    "pilot_throttle" REAL,
    "edit_angle" REAL, --Manually edited
    "edit_throttle" REAL, --Manually edited
    "deleted" BOOLEAN NOT NULL DEFAULT false CHECK (deleted IN (0, 1)),
    PRIMARY KEY(source_id, tub_index),
    FOREIGN KEY(source_id) REFERENCES Sources (source_id)
       ON UPDATE CASCADE
       ON DELETE RESTRICT
    );
sqlite> .schema Sources
CREATE TABLE IF NOT EXISTS "Sources" (
    "source_id" INTEGER PRIMARY KEY NOT NULL,
    "name" TEXT NOT NULL,
    "full_path" TEXT NOT NULL,
    "version" REAL DEFAULT 2.0, -- DonkeyCar Tub version
    "image_width" INTEGER,
    "image_height" INTEGER,
    "created" REAL,
    "count" INTEGER,
    "inputs" TEXT,
    "types" TEXT,
    UNIQUE(name, full_path)
    );
"""

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
class SqliteFormat(DriveFormat):
    """ A class to represent an sqlite data that holds DonkeyCar data.

        Current assumptions:
            We only care about editing steering and throttle.
            Steering and throttle should be clipped to -1/1.
    """

    source_tag = "sqlite.source: " # Tag to add to the 'path' so we know it's an sqlite file
    connection = None # Database connection object

    @classmethod
    def setConnection(cls, connection):
        cls.connection = connection

    def __init__( self, path ):
        DriveFormat.__init__(self)

        if SqliteFormat.connection is None:
            raise Exception("SqliteFormat error: No database connection")

        if not path.startswith(self.source_tag):
            raise IOError( "SqliteFormat error. Not a valid source id: {}".format( path ) )

        try:
            print( f"SqliteFormat: {path[len(self.source_tag):]}" )
            self.source_id = int(path[len(self.source_tag):])
        except ValueError:
            raise IOError( "SqliteFormat error. Not a valid source id: {}".format( path ) )

        print( f"SqliteFormat: source_id = {self.source_id}" )
        self.meta = self.getMetaData()
        self.deleted_indexes = []
        self.edit_list = set()
        self.shape = None

    def getMetaData(self):
        # add all meta data from the Sources and TubMetadata tables to a dictionary
        # Possibly use 'conn.row_factory = sqlite3.Row' to get a dictionary with column names instead of a tuple
        get_source_sql = """SELECT name, full_path, version, image_width, image_height, created, count, inputs, types
                            FROM Sources WHERE source_id = ?"""
        get_meta_sql = """SELECT key, value FROM SourceMeta WHERE source_id = ?"""

        meta = {}
        cursor = SqliteFormat.connection.cursor()
        cursor.execute(get_source_sql, (self.source_id,))

        if cursor.rowcount == 0:
            raise IOError( "SqliteFormat error. No source found for id: {}".format( self.source_id ) )

        if cursor.rowcount > 1:
            print( f"Warning: More than one source returned for source_id {self.source_id}" )

        for row in cursor:
            meta = dict(row)

        cursor.execute(get_meta_sql, (self.source_id,))
        for row in cursor:
            meta[row[0]] = row[1]

        return meta

    def _load( self, image_norm=True, progress=None ):
        get_training_sql = f"""SELECT tub_index, image_path, user_angle, user_throttle, pilot_angle, pilot_throttle,
                                edit_angle, edit_throttle, deleted,
                                Sources.full_path || '/' || '{Tub.images()}' || '/' || TubRecords.image_path as "cam/image_array"
                                FROM TubRecords, Sources
                               WHERE TubRecords.source_id = ?
                                 AND TubRecords.source_id = Sources.source_id
                            ORDER BY tub_index;"""

        records = []
        images = [] # Store images separately

        cursor = SqliteFormat.connection.cursor()
        cursor.execute(get_training_sql, (self.source_id,))
        total = cursor.rowcount

        count = 0
        for row in cursor:
            img_path = row['cam/image_array']
            try:
                img = Image.open( img_path )
                img_arr = np.asarray(img)
                if self.shape is None:
                    self.shape = img_arr.shape
            except Exception as ex:
                print( f"Failed to load image: {img_path}" )
                print( f"   Exception: {ex}" )
            records.append( row )
            images.append(img_arr)
            count += 1
            progress(count, total)

        self.records = records
        self.images = images

    def load( self, progress=None ):
        self._load(progress=progress)
        self.setClean()

    def save( self ):
        if self.isClean():
            return

        # loop through the edit list and make an array of values
        # then update the database in one go
        edits = []
        for ix in self.edit_list:
            rec = self.records[ix]
            edits.append( (rec['edit_angle'], rec['edit_throttle'], self.source_id, rec['tub_index']) )

        print( f"SqliteFormat edits: {edits}" )
        update_edited = f"""UPDATE TubRecords SET edit_angle = ?, edit_throttle = ? WHERE source_id = ? AND tub_index = ?"""
        cursor = SqliteFormat.connection.cursor()
        cursor.executemany(update_edited, edits)
        SqliteFormat.connection.commit()

        self.edit_list.clear()
        self.setClean()

    def count( self ):
        return len(self.records)

    def imageForIndex( self, index ):
        img = self.images[index]
        if self.isIndexDeleted(index):
# This grayed out image ends up looking ugly, can't figure out why
            tmp = img.mean(axis=-1,dtype=img.dtype,keepdims=False)
            tmp = np.repeat( tmp[:,:,np.newaxis], 3, axis=2 )
            return tmp
        return img

    def get_angle_throttle(self, record):
        modes = ['user', 'pilot', 'edit']

        record = dict(record)
        angle = 0.0
        throttle = 0.0

        for mode in modes:
            m_angle = record.get( mode + '_angle', None )
            m_throttle = record.get( mode + '_throttle', None )
            if m_angle is not None and (m_angle != 0.0 or mode == 'edit'):
                angle = m_angle
            if m_throttle is not None and (m_throttle != 0.0 or mode == 'edit'):
                throttle = m_throttle

        return angle, throttle

    def actionForIndex( self, index ):
        rec = self.records[index]
        angle, throttle = self.get_angle_throttle(rec)
        return [angle, throttle]

    def setActionForIndex( self, new_action, index ):
        rec = self.records[index]
        angle, throttle = self.get_angle_throttle(rec)
        old_action = [angle, throttle]
        if not np.array_equal( old_action, new_action ):
            #print( f"setActionForIndex" )
            #print( f"   user: {rec['user_angle']}/{rec['user_throttle']}" )
            #print( f"    new: {new_action}" )

            if type(rec) is not dict:
                rec = dict(rec)
                self.records[index] = rec

            rec["edit_angle"] = new_action[0]
            rec["edit_throttle"] = new_action[1]
            self.edit_list.add(index)
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
        if cls.connection is None:
            raise RuntimeError( "No database connection" )

        if path.startswith(cls.source_tag):
            return True

        return False

    @staticmethod
    def defaultInputTypes():
        return [{"name":"Images", "type":"numpy image", "shape":(120,160,3)}]

    def inputTypes(self):
        res = SqliteFormat.defaultInputTypes()
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
DriveFormat.registerFormat( "SqliteFormat", SqliteFormat )

def runTests(args):
    DriveFormat.testFormat( SqliteFormat, args.file[0], [2.0, -5.5] )
    d = SqliteFormat(args.file[0])
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
