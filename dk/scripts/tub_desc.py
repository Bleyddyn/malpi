"""
Describe Tub files found in one or more directories.

Usage:
    tub_desc.py [--file=<file> ...] [--stats] [--tod] [<dir>...]


Options:
    -h --help        Show this screen.
    <dir>            One or more directories that are Tub files, or contain Tub files.
    -f --file=<file> A text file containing paths to tub files, one per line. Option may be used more than once.
    -s --stats       Calculate mean and standard deviation for throttle. [Default: False]
    --tod            Display time of day for beginning of each tub file. [Default: False]

"""
import os
import time
from docopt import docopt
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

def describe_tub( tub_dir, stats=False, time_of_day=False ):
    """ TODO: This should be generalized to return only user-requested meta data.
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
        for rec in recs:
            with open(rec, 'r') as fp:
                json_data = json.load(fp)
            angle, one_thr = Tub.get_angle_throttle(json_data)
            thr.append( one_thr )
        thr_m = np.mean(thr)
        thr_v = np.std(thr)
        st = "\t{:0.3}/{:0.3}".format( float(thr_m), float(thr_v) )
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
    print( "{}\t{}\t{}\t{}\t{}\t{}\t{}{}{}".format( os.path.basename(tub_dir), count, loc, task, driver, throttle, steering, st, tod ) )
    return count

def check_path( apath, counter=None, stats=False, time_of_day=False ):
    if os.path.isdir(apath):
        meta_path = os.path.join( apath, "meta.json" )
        if os.path.exists(meta_path):
            cnt = describe_tub(apath, stats, time_of_day)
            if counter is not None:
                counter.append( cnt )
            return True
    return False

if __name__ == "__main__":
    args = docopt(__doc__)

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")

    do_stats = args['--stats']
    do_tod = args['--tod']
    dirs = preprocessFileList( args['--file'] )
    dirs.extend( args['<dir>'] )
    dirs = [os.path.expanduser(adir) for adir in dirs]

    done = []
    counts = []
    if do_stats:
        stat_str = "\tMean/Std"
    else:
        stat_str = ""
    if do_tod:
        tod_header = "\tTime"
    else:
        tod_header = ""
    print( "\nTub\t# Samples\tLocation\tTask\tDriver\tThrottle\tSteering{}{}".format( stat_str, tod_header ) )
    for adir in sorted(dirs):
        if adir not in done and os.path.isdir(adir):
            if not check_path(adir, counter=counts, stats=do_stats, time_of_day=do_tod):
                for afile in sorted(os.listdir(adir)):
                    fpath = os.path.join(adir, afile)
                    if fpath not in done:
                        check_path( fpath, counter=counts, stats=do_stats, time_of_day=do_tod )
                        done.append(fpath)
            else:
                done.append(adir)


    print()
    print( "Total samples: {}".format( np.sum(counts) ) )
