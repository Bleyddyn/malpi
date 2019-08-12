""" Copy AI/pilot generated data in Tub records into user fields.
    The tub file can then be used as normal for training or analysis.
    If data already exists in the user fields it is first copied to 'orig'
        fields. This step probably isn't necessary since we only copy
        pilot data if the user values are null or zero.
"""
import os
import sys
import json
import argparse
from donkeycar.parts.datastore import Tub
from donkeycar.utils import get_record_index

    
def pilot2user( path ):
    tub = Tub(path)
    tub.exclude.clear() # we want to do excluded records, just in case they get un-excluded

    records = tub.gather_records()
    for rec_path in records:
        try:
            with open(rec_path, 'r') as fp:
                rec = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record')
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        pa = rec.get('pilot/angle', None)
        pt = rec.get('pilot/throttle', None)
        ua = rec.get('user/angle', 0.0)
        ut = rec.get('user/throttle', 0.0)

        if (ua != 0.0) or (ut != 0.0):
            #idx = get_record_index(rec_path)
            print( "Already have user values for {}".format( rec_path ) )
        else:
            """ copy user to orig if not done already, copy pilot to user """
            if pa is not None:
                if 'orig/angle' not in rec and ua is not None:
                    rec['orig/angle'] = ua
                rec['user/angle'] = pa
            if pt is not None:
                if 'orig/throttle' not in rec and ut is not None:
                    rec['orig/throttle'] = ut
                rec['user/throttle'] = pt

            print( "Writing record {}".format( rec_path ) )
            with open(rec_path, 'w') as fp:
                json.dump(rec, fp)

def getOptions():

    parser = argparse.ArgumentParser(description='Copy AI Pilot outputs to User outputs in a Tub file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', nargs=1, metavar="File", help='Tub data file to open')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = getOptions()

    pilot2user( args.file[0] )
