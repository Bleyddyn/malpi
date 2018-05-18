import os
import argparse
import json
import numpy as np

def loadActions( drive_dir, indent="" ):
    ifname = os.path.join( drive_dir, 'image_actions.npy' )
    if os.path.exists(ifname):
        actions = np.load(ifname)
    else:
        print( "{}Missing image_actions.npy".format( indent ) )
        actions = None

    return actions

def saveAuxMeta( drive_dir, indent="" ):
    auxMeta = {}
    aux_file = os.path.join( drive_dir, "aux.json" )
    if os.path.exists(aux_file):
        with open(aux_file,'r') as f:
            auxMeta = json.load(f)

    cmeta = { "name": "continuous", "type": "continuous", "default": 0.0, "count": 2, "min": -1.0, "max":1.0}
    auxMeta["Continuous"] = cmeta

    with open(aux_file,'w') as f:
        json.dump(auxMeta,f)

def saveActions( drive_dir, cactions, indent="" ):
    auxFile = os.path.join( drive_dir, "{}_aux.npy".format( "continuous" ) )
    np.save( auxFile, np.array(cactions) )
    saveAuxMeta( drive_dir )

def runTests(args, data):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='View or post-process collected drive data.')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
#    group.add_argument('--extract', help='Extract images from original file(s) into a new file with this prefix')

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            tmp_dirs = f.read().split('\n')
            args.dirs.extend(tmp_dirs)

    if len(args.dirs) == 0 and not args.test_only:
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    for i in reversed(range(len(args.dirs))):
        if args.dirs[i].startswith("#"):
            del args.dirs[i]
        elif len(args.dirs[i]) == 0:
            del args.dirs[i]
            
    return args

if __name__ == "__main__":
    args = getOptions()

    indent="  "

    if args.test_only:
        runTests(args)
        exit()

    for adir in args.dirs:
        print( adir )

# Load image_actions
# convert to continuous controls
# write out to an auxiliary file
# Add meta info to aux.json

        auxFile = os.path.join( adir, "{}_aux.npy".format( "continuous" ) )
        if os.path.exists(auxFile):
            print( "   Continuous action file already exists: {}".format( auxFile ) )
            continue

        actions = loadActions( adir, indent=indent )
        cactions = []
        speed = 0.5
        last_act = (0.0,0.0)
        if actions is None:
            print( "  Missing or empty actions file.")
            continue

        for act in actions:
            if act == "stop":
                last_act = (0.0,0.0)
            elif act == "forward":
                last_act = (speed,speed)
            elif act == "backward":
                last_act = (-speed,-speed)
            elif act == "left":
                last_act = (-speed,speed)
            elif act == "right":
                last_act = (speed,-speed)
            elif act.startswith( "speed" ):
                speed = float(act[6:]) / 255.0
            else:
                raise "Invalid action: {}".format( act )
            cactions.append( last_act )

        saveActions( adir, cactions, indent=indent )
