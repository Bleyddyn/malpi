"""
Describe Tub files found in one or more directories.
"""
import os
import time
import argparse
import json
from datetime import datetime

import numpy as np
import donkeycar as dk
from donkeycar.parts.datastore import Tub

def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessFileList( filelist ):
    dirs = []
    if filelist is not None:
        for afile in filelist:
            afile = os.path.expanduser(afile)
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    removeComments( dirs )
    return dirs

def describe_tub( tub_dir, stats=False, time_of_day=False, meta=[], img=False ):
    """ TODO: This should be generalized to return only user-requested meta data.
        TODO: Add a check for image sizes
    """
    tub = Tub(tub_dir)
    count = tub.get_num_records()
    loc = tub.meta.get("location", "NA")
    task = tub.meta.get("task", "NA")
    driver = tub.meta.get("driver", "NA")
    tod = tub.meta.get("start", None)
    throttle = tub.meta.get("JOYSTICK_MAX_THROTTLE", "NA")
    steering = tub.meta.get("JOYSTICK_STEERING_SCALE", "NA")
    if stats:
        recs = tub.gather_records()
        thr = []
        reward = []
        for rec in recs:
            with open(rec, 'r') as fp:
                json_data = json.load(fp)
            angle, one_thr = Tub.get_angle_throttle(json_data)
            thr.append( one_thr )
            if 'sim/reward' in json_data:
                reward.append( json_data['sim/reward'] )
            elif 'sim/info' in json_data:
                reward.append( json_data['sim/info']['reward'] )
        thr_m = np.mean(thr)
        thr_v = np.std(thr)
        st = "\t{:0.3}/{:0.3}".format( float(thr_m), float(thr_v) )
        if len(reward) > 0:
            st = "\t{:0.3}/{:0.3}/{:0.3}/{:0.3}".format( float(np.min(reward)), float(np.mean(reward)), float(np.std(reward)), float(np.max(reward)) )
        else:
            st = "\tNA"
    else:
        st = ""

    if time_of_day:
        if tod is not None:
            tod = int(tod)
            tod = datetime.fromtimestamp(tod).strftime('\t%H:%M')
        else:
            tod = "\t"
    else:
        tod = ""

    meta_st = ""
    for key in meta:
        if key in tub.inputs:
            meta_st += "\tInput"
        elif key in tub.meta:
            meta_st += "\t{}".format( tub.meta[key] )
        else:
            meta_st += "\tNo"

    img_st = ""
    if img:
        data = tub.get_record(1)
        img_array = data['cam/image_array']
        img_st = "\t{}".format( img_array.shape )

    print( "{}\t{}\t{}\t{}\t{}\t{}\t{}{}{}{}{}".format( os.path.basename(tub_dir), count, loc, task, driver, throttle, steering, st, tod, meta_st, img_st ) )
    return count

def check_path( apath, counter=None, stats=False, time_of_day=False, meta=[], img=False ):
    if os.path.isdir(apath):
        meta_path = os.path.join( apath, "meta.json" )
        if os.path.exists(meta_path):
            cnt = describe_tub(apath, stats, time_of_day, meta=meta, img=img)
            if counter is not None:
                counter.append( cnt )
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Describe Tub files found in one or more directories.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--stats', action='store_true', default=False, help='Calculate mean and standard deviation for throttle.')
    parser.add_argument('--tod', action='store_true', default=False, help='Display time of day for beginning of each tub file.')
    parser.add_argument('--img', action='store_true', default=False, help='Display image size.')
    parser.add_argument('-f', '--file', nargs="*", help='A text file containing paths to tub files, one per line. Option may be used more than once.')
    parser.add_argument('dir', nargs='*', help='One or more directories that are Tub files, or contain Tub files.')
    parser.add_argument('--meta', nargs='*', default=[], help='One or more meta keys to search for in Tub files.')

    args = parser.parse_args()

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")

    dirs = preprocessFileList( args.file )
    dirs.extend( args.dir )
    dirs = [os.path.expanduser(adir) for adir in dirs]

    done = []
    counts = []
    if args.stats:
        stat_str = "\tMean/Std"
    else:
        stat_str = ""
    if args.tod:
        tod_header = "\tTime"
    else:
        tod_header = ""
    meta_header = ""
    for meta in args.meta:
        meta_header += "\t" + meta

    img_header = ""
    if args.img:
        img_header = "\tImg Size"

    print( "\nTub\t# Samples\tLocation\tTask\tDriver\tThrottle\tSteering{}{}{}{}".format( stat_str, tod_header, meta_header, img_header ) )
    for adir in sorted(dirs):
        if adir not in done and os.path.isdir(adir):
            if not check_path(adir, counter=counts, stats=args.stats, time_of_day=args.tod, meta=args.meta, img=args.img):
                for afile in sorted(os.listdir(adir)):
                    fpath = os.path.join(adir, afile)
                    if fpath not in done:
                        check_path( fpath, counter=counts, stats=args.stats, time_of_day=args.tod, meta=args.meta, img=args.img)
                        done.append(fpath)
            else:
                done.append(adir)


    print()
    print( "Total samples: {}".format( np.sum(counts) ) )
