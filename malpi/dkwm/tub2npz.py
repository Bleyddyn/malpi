""" Read DonkeyCar Tub files and write each one to a compressed numpy file (.npz).
    Output file will have the same name and location as the tub, with a .npz extension.
    Output file will have keys for 'images' (np.int8), 'actions', 'dt_stamps' (unix epoch).
    Output file may also have a 'lanes' key for lane auxiliary labels.
"""

import os
import argparse
import json
import datetime

import donkeycar as dk
from donkeycar.templates.train import preprocessFileList

from collections import namedtuple
import json
import numpy as np
from donkeycar.parts.datastore import Tub
from donkeycar.utils import get_record_index
from PIL import Image

ImageDim = namedtuple('ImageDim', ['IMAGE_W', 'IMAGE_H', 'IMAGE_DEPTH'])

def load_image_arr(filename, img_dim):
    try:
        img = Image.open(filename)
        if img.height != img_dim.IMAGE_H or img.width != img_dim.IMAGE_W:
            img = img.resize((img_dim.IMAGE_W, img_dim.IMAGE_H))
        img_arr = np.array(img)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr

# Load one Tub at a time, build a numpy array for use in training and save it back to the tub dir
def load_one_tub( tub_path, img_dim, progress=None ):
    tub = Tub(tub_path)

    if not hasattr( tub, 'start_time' ):
        print( "   Tub is missing start time, skipping: {}".format( tub_path ) )
        return {}

    records = tub.gather_records()
    images = []
    actions = []
    dt_stamps = []
    lanes = [] # TODO Generalize this so it will read all auxiliary labels
    count = len(records)
    if progress is not None:
        progress(0, count)

    for idx, record_path in enumerate(records):
        if progress is not None:
            progress(idx, count)
        #if idx % 100 == 0:
        #    print( "Record {} of {}".format( idx, count ))

        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except Exception as ex:
            print( "Failed to load {}".format( record_path ) )
            print( "  Exception: {}".format( ex ) )
            continue

        if 'milliseconds' in json_data:
            ms = tub.start_time + (json_data['milliseconds'] / 1000.0)
        else:
            index = get_record_index(record_path)
            ms = tub.start_time + (index / 20.0)

        basepath = os.path.dirname(record_path)
        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)
        image = load_image_arr(image_path, img_dim)
        angle, throttle = Tub.get_angle_throttle(json_data)

        images.append( image )
        actions.append( [angle, throttle] )
        dt_stamps.append( ms )
        if "lanes" in json_data:
            lanes.append( json_data["lanes"] )

    ret = { "images": np.array(images, dtype=np.uint8), "actions": np.array(actions, dtype=np.float32),
            "dt_stamps": np.array(dt_stamps, dtype=np.float32) }
    if len(lanes) > 0:
        ret["lanes"] = np.array(lanes)
    return ret

def tubs_to_npz( dirs, img_dim, overwrite=False, verbose=True, progress=None ):
    """ dirs: an iterable of path strings to one or more tub files
        img_dim: DonkeyCar config or ImageDim named tuple with image width and height
        verbose: display sample counts for each tub
        progress: a function that takes index and count as each tub is loaded
    """
    for tub in dirs:
        if overwrite or not os.path.exists( tub + ".npz" ):
            print( "Loading {}".format( tub ) )
            data = load_one_tub(tub, img_dim, progress)
            if len(data) > 0:
                if verbose:
                    print( "   Images: {}".format( data["images"].shape ) )
                    print( "   Actions: {}".format( data["actions"].shape ) )
                    print( "   Dates: {}".format( data["dt_stamps"].shape ) )
                    if "lanes" in data:
                        print( "   Lanes: {}".format( data["lanes"].shape ) )
                tub += ".npz"
                np.savez_compressed(tub, **data)
            print("")

def cmd_progress( idx, count ):
    if idx % 100 == 0:
        print( "   Record {} of {}".format( idx, count ))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load tub files and resave as compressed numpy.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--overwrite', action="store_true", default=False, help='Overwrite existing npz files.')
    parser.add_argument('file', nargs='+', help='Text file with a list of tubs to convert.')

    args = parser.parse_args()

    dirs = preprocessFileList( args.file )

    try:
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py") # retry in the current directory

    img_dim = ImageDim(cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)

    tubs_to_npz( dirs, img_dim, overwrite=args.overwrite, progress=cmd_progress )
