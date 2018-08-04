""" Functions for adding standard arguments to an argparse object,
    and for handling the standard malpi list of drive directories.
"""

import argparse

def addMalpiOptions( parser, test_only=True):

    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', action="append", help='File with one directory per line')
    if test_only:
        parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessOptions( args ):
    if args.file is not None:
        for afile in args.file:
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                args.dirs.extend(tmp_dirs)

    removeComments( args.dirs )

    if hasattr( args, 'val' ):
        if args.val is not None:
            with open(args.val, "r") as f:
                tmp_dirs = f.read().split('\n')
                args.val = tmp_dirs
            removeComments( args.val )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test MaLPi Options.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    addMalpiOptions( parser )
    args = parser.parse_args()
    preprocessOptions(args)

    for fname in args.dirs:
        print( fname )
