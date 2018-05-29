import os
import argparse
import json
import pickle
import numpy as np

def loadDrive( drive_dir, indent="" ):
    ifname = os.path.join( drive_dir, 'drive.pickle' )
    if not os.path.exists(ifname):
        print( "{}No drive.pickle. Possibly an older drive format".format( indent ) )
        return None

    with open(ifname,'r') as f:
        data = pickle.load(f)

    return data

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

def convertFromCategorical( drive_dir, indent="" ):
# Load image_actions
# convert to continuous controls
# write out to an auxiliary file
# Add meta info to aux.json

    auxFile = os.path.join( drive_dir, "{}_aux.npy".format( "continuous" ) )
    if os.path.exists(auxFile):
        print( "   Continuous action file already exists: {}".format( auxFile ) )
        return

    actions = loadActions( drive_dir, indent=indent )
    cactions = []
    speed = 0.5
    last_act = (0.0,0.0)
    if actions is None:
        print( "  Missing or empty actions file.")
        return

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

    saveActions( drive_dir, cactions, indent=indent )

# Use this to modify long runs of 0.5/0.5 values in continuous aux files
# Also add some small gaussian variability to each value other than 0.0 and 1.0
def rle(inarray):
    """ From: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

def fixContinuousActions( drive_dir, indent="" ):
    """ load image_times, action_times and actions
        get first image time
        step through action times until the first one after image time
        go back one action (or average actions between this image and previous?)
    """
    data = loadDrive( drive_dir, indent=indent )
    image_times = data['image_times']
    image_actions = data['image_actions']
    action_times = data['action_times']
    actions = data['actions']

    ia_ext = loadActions( drive_dir, indent=indent )

    size = (120,120)
    basename = "images_{}x{}".format( size[0], size[1] )
    im_file = os.path.join( drive_dir, basename+".npy" )
    images = np.load(im_file)

    if len(image_times) != images.shape[0]:
        print( "{}Internal/external image sizes don't match: {} {}".format( indent, len(image_times), images.shape ) )

    print( "{}Internal/external image actions: {} {}".format( indent, len(image_actions), ia_ext.shape ) )
    if ia_ext.shape[0] != len(image_actions):
        print( "{}Internal/external image actions don't match: {} {}".format( len(image_actions), ia_ext.shape[0] ) )
        return

    if len(image_times) == len(image_actions):
        print( "{}Image times and image actions match. No fix necessary".format( indent ) )
        return

    if len(action_times) != len(actions):
        print( "{}Action times and actions do not match.".format( indent ) )
        return

    new_actions = []
    idx = 0
    last_action = (0.0,0.0)
    for it in image_times:
        while it > action_times[idx]:
            last_action = actions[idx]
            idx += 1
            continue
        new_actions.append(last_action)
        last_action = actions[idx]
        idx += 1
        if idx >= len(actions):
            print( "{}Ran out of actions.".format( indent ) )
            return

    print( "{}Images/actions: {} {}".format( len(image_times), len(new_actions) ) )
        

def runTests(args, data):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Post-process continuous-action drive data.')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
    group.add_argument('--convert', action="store_true", default=False, help='Convert a categorical drive to continuous, saving in an auxiliary data file')
    group.add_argument('--fix', action="store_true", default=False, help='Fix a continuous drive that has mismatched image/action data')

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

    for drive_dir in args.dirs:
        print( drive_dir )

        if args.convert:
            convertFromCategorical( drive_dir, indent="   " )

        if args.fix:
            fixContinuousActions( drive_dir, indent="   " )
