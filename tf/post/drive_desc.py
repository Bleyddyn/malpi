import os
import pickle
import argparse
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

def dumpArrayToFile( data, key, fname=None, indent="" ):
    if not fname:
        fname = key + ".pickle"
    if not key in data:
        print( "{}Missing '{}' key in drive data.".format( indent, key ) )
        return False

    array = data[key]
    with open(fname, 'w') as f:
        pickle.dump( array, f, protocol=pickle.HIGHEST_PROTOCOL )
    return True

def readArrayFromFile( key ):
    fname = key + ".pickle"
    with open(fname, 'r') as f:
        data = pickle.load( f )
    return data

def getImagesFrom( filename ):
    with open(filename,'r') as f:
        data = pickle.load(f)

    small = []
    for img in data:
        img = imresize(img, (120,120), interp='nearest' )
        small.append(img)
    return small

def loadDrive( drive_dir, indent="" ):
    ifname = os.path.join( drive_dir, 'drive.pickle' )
    if not os.path.exists(ifname):
        print( "{}No drive.pickle. Possibly an older drive format".format( indent ) )
        return None

    with open(ifname,'r') as f:
        data = pickle.load(f)

    return data

def makeActions( drive_dir, data, indent="" ):
    ofname = os.path.join( drive_dir, 'image_actions.pickle' )
    if not os.path.exists(ofname):
        if dumpArrayToFile( data, 'image_actions', fname=ofname ):
            print( "{}Saved {} Image actions".format( indent, len(data['image_actions'] ) ) )
    else:
        print( "{}Image actions already exists".format( indent ) )

def makeSmallImages( drive_dir, data, indent="" ):
    ofname = os.path.join( drive_dir, 'images_120x120.pickle' )
    if not os.path.exists(ofname):
        smallImages = []
        for fname in os.listdir(drive_dir):
            if fname.startswith("images_") and fname.endswith(".pickle"):
                ifname = os.path.join( drive_dir, fname )
                print( "{}Reading {}".format( indent, fname ) )
                im1 = getImagesFrom(ifname)
                smallImages.extend(im1)

        if len(smallImages) == 0 and 'images' in data:
            img_data = data['images']
            for img in img_data:
                img = imresize(img, (120,120), interp='nearest' )
                smallImages.append(img)

        print( "{}Writing {} small images".format( indent, len(smallImages) ) )

        with open(ofname,'wb') as f:
            pickle.dump( smallImages, f, protocol=pickle.HIGHEST_PROTOCOL )
    else:
        print( "{}Small Images file already exists".format( indent ) )

def writeToSeparateFiles():
    for key in ["action_times", "image_actions", "actions", "image_times", "images"]:
        dumpArrayToFile( data, key )

    with open( "meta.txt", 'w' ) as mf:
        mf.write( "Date: {}\n".format( str(data["date"]) ) )
        mf.write( "Model: {}\n".format( str(data["model"]) ) )

def readFromSeparateFiles():
    for key in ["action_times", "image_actions", "actions", "image_times", "images"]:
        try:
            data = readArrayFromFile( key )
            print( "{}: {}".format( key, len(data) ) )
        except:
            print( "{} Failed".format( key ) )

def describeKey( key, key_data, indent="" ):
    if isinstance(key_data, basestring):
        print( "{}{}: {}".format( indent, key, key_data ) )
    else:
        key_len = 0
        try:
            key_len = len(key_data)
        except:
            pass
        if key_len > 0:
            print( "{}{} count: {}".format( indent, key, len(key_data) ) )
        else:
            print( "{}{}: {}".format( indent, key, str(key_data) ) )

def describeData( data, indent="" ):
    for key in data.keys():
        describeKey( key, data[key], indent=indent )

def sampleImages( drive_dir, data, indent="", count=5 ):

    images = data['images']
    with open(os.path.join( drive_dir, 'images_120x120.pickle' ),'r') as f:
        small_images = pickle.load(f)

    print( "Image array lengths (S/L): {}/{}".format( len(small_images), len(images) ) )

    if len(images) == 0 or (len(images) != len(small_images)):
        print( "Mismatched image array lengths" )
        return

    for i in range(count):
        idx = np.random.randint(0,len(images))

        img_large = images[idx]
        img_small = small_images[idx]

        plt.figure(1,figsize=(16, 18), dpi=80)
        ax1=plt.subplot(211)
        ax1.imshow(img_large)
        ax1=plt.subplot(212)
        ax1.imshow(img_small)

        plt.show()


def runTests(args, data):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='View or post-process collected drive data.')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
    parser.add_argument('--desc', action="store_true", default=False, help='Describe each drive, then exit')
    parser.add_argument('--sample', action="store_true", default=False, help='Display sample images, then exit')

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            tmp_dirs = f.read().split('\n')
            args.dirs.extend(tmp_dirs)

    if len(args.dirs) == 0 and not args.test_only:
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    return args

if __name__ == "__main__":
    args = getOptions()

    indent="  "

    if args.test_only:
        runTests(args)
        exit()

    if args.sample:
        data = loadDrive( args.dirs[0], indent=indent )
        if data is not None:
            sampleImages( args.dirs[0], data, indent=indent )
        exit()
        
    for adir in args.dirs:
        print( adir )
        data = loadDrive( adir, indent=indent )

        if data is not None:
            if args.desc:
                describeData( data, indent=indent )
            else:
                if not 'model' in data:
                    print( "{}No model name in drive.pickle. Possibly an older drive format".format( indent ) )
                else:
                    print( "{}Model: {}".format( indent, data['model'] ) )
                makeActions( adir, data, indent=indent )
                makeSmallImages( adir, data, indent=indent )
