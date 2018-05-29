import os
import pickle
import argparse
import re
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
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

def getImagesFrom( filename, size=(120,120) ):
    with open(filename,'r') as f:
        data = pickle.load(f)

    small = []
    for img in data:
        img = imresize(img, size, interp='nearest' )
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

def convertDriveToNumpy( drive_dir, size=(120,120), indent="" ):
    ifname = os.path.join( drive_dir, 'image_actions.pickle' )
    if os.path.exists(ifname):
        with open(ifname,'r') as f:
            actions = pickle.load(f)
            actions = np.array(actions)
            actions = actions.astype('str')
            ofname = os.path.join( drive_dir, 'image_actions.npy' )
            np.save(ofname, actions)
            print( "{}Converted image_actions to numpy format".format( indent ) )
    else:
            print( "{}Missing image_actions.pickle".format( indent ) )
    # repeat for images_120x120
    basename = 'images_{}x{}'.format( size[0], size[1] )
    ifname = os.path.join( drive_dir, basename+'.pickle' )
    if os.path.exists(ifname):
        with open(ifname,'r') as f:
            images = pickle.load(f)
            images = np.array(images)
            ofname = os.path.join( drive_dir, basename+'.npy' )
            np.save(ofname, images)
            print( "{}Converted {} to numpy format".format( indent, basename ) )
    else:
            print( "{}Missing {}.pickle".format( indent, basename ) )

def makeActions( drive_dir, data, indent="" ):
    ofname = os.path.join( drive_dir, 'image_actions.pickle' )
    if not os.path.exists(ofname):
        if dumpArrayToFile( data, 'image_actions', fname=ofname ):
            print( "{}Saved {} Image actions".format( indent, len(data['image_actions'] ) ) )
    else:
        print( "{}Image actions already exists".format( indent ) )

def makeSmallImages( drive_dir, data, size=(120,120), indent="" ):
    ofname = os.path.join( drive_dir, 'images_{}x{}.pickle'.format( size[0], size[1] ) )
    if not os.path.exists(ofname):
        smallImages = []
        image_files = []
        for fname in os.listdir(drive_dir):
            if fname.startswith("images_") and fname.endswith(".pickle"):
                image_files.append(fname)
        image_files.sort(key=natural_keys)
        for fname in image_files:
            ifname = os.path.join( drive_dir, fname )
            print( "{}Reading {}".format( indent, fname ) )
            im1 = getImagesFrom(ifname, size=size)
            smallImages.extend(im1)

        if len(smallImages) == 0 and 'images' in data:
            img_data = data['images']
            for img in img_data:
                img = imresize(img, size, interp='nearest' )
                smallImages.append(img)

        print( "{}Writing {} small images".format( indent, len(smallImages) ) )

        with open(ofname,'wb') as f:
            pickle.dump( smallImages, f, protocol=pickle.HIGHEST_PROTOCOL )
    else:
        print( "{}Small Images file already exists".format( indent ) )

def extractNewImages( drive_dir, data, prefix, size=(120,120), indent="" ):
    ofname = os.path.join( drive_dir, '{}_{}x{}.npy'.format( prefix, size[0], size[1] ) )
    if not os.path.exists(ofname):
        smallImages = []
        image_files = []
        for fname in os.listdir(drive_dir):
            if fname.startswith("images_") and fname.endswith(".pickle"):
                image_files.append(fname)
        image_files.sort(key=natural_keys)
        for fname in image_files:
            ifname = os.path.join( drive_dir, fname )
            print( "{}Reading {}".format( indent, fname ) )
            im1 = getImagesFrom(ifname, size=size)
            smallImages.extend(im1)

        if len(smallImages) == 0 and 'images' in data:
            img_data = data['images']
            for img in img_data:
                img = imresize(img, size, interp='nearest' )
                smallImages.append(img)

        print( "{}Writing {} small images".format( indent, len(smallImages) ) )

        images = np.array(smallImages)
        np.save(ofname, images)
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

def imageStats( drive_dir, indent="" ):
    with open(os.path.join( drive_dir, 'images_120x120.pickle' ),'r') as f:
        small_images = pickle.load(f)
    print( "{}Small Images min/mean/std/max: {} / {} / {} / {}".format( indent, np.min(small_images), np.mean(small_images), np.std(small_images), np.max(small_images) ) )
    small_images = np.array(small_images)
    small_images = small_images.astype(np.float)
    small_images[:,:,:,0] -= np.mean(small_images[:,:,:,0])
    small_images[:,:,:,1] -= np.mean(small_images[:,:,:,1])
    small_images[:,:,:,2] -= np.mean(small_images[:,:,:,2])

    print( "{}Red min/mean/std/max: {} / {} / {} / {}".format( indent, np.min(small_images[:,:,:,0]), np.mean(small_images[:,:,:,0]), np.std(small_images[:,:,:,0]), np.max(small_images[:,:,:,0]) ) )
    print( "{}Green min/mean/std/max: {} / {} / {} / {}".format( indent, np.min(small_images[:,:,:,1]), np.mean(small_images[:,:,:,1]), np.std(small_images[:,:,:,1]), np.max(small_images[:,:,:,1]) ) )
    print( "{}Blue min/mean/std/max: {} / {} / {} / {}".format( indent, np.min(small_images[:,:,:,2]), np.mean(small_images[:,:,:,2]), np.std(small_images[:,:,:,2]), np.max(small_images[:,:,:,2]) ) )

    print( "{}Shape: {}".format( indent, small_images.shape ) )
    print( "{}Type: {}".format( indent, small_images.dtype ) )

def describeKey( key, key_data, indent="", do_print=True ):
    if isinstance(key_data, basestring):
        output = "{}{}: {}".format( indent, key, key_data )
    else:
        key_len = 0
        try:
            key_len = len(key_data)
        except:
            pass
        if key_len > 0:
            output = "{}{} count: {}".format( indent, key, len(key_data) )
        else:
            output = "{}{}: {}".format( indent, key, str(key_data) )

    if do_print:
        print( output )

    return output

def describeData( data, indent="", do_print=True ):
    output = ""
    for key in data.keys():
        output += describeKey( key, data[key], indent=indent, do_print=do_print ) + "\n"
    return output

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

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')
    group.add_argument('--desc', action="store_true", default=False, help='Describe each drive, then exit')
    group.add_argument('--sample', action="store_true", default=False, help='Display sample images, then exit')
    group.add_argument('--stats', action="store_true", default=False, help='Display image stats for each directory')
    group.add_argument('--py2py3', action="store_true", default=False, help='Read pickle files and save to numpy files')
    group.add_argument('--extract', help='Extract images from original file(s) into a new file with this prefix')

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

    if args.sample:
        data = loadDrive( args.dirs[0], indent=indent )
        if data is not None:
            sampleImages( args.dirs[0], data, indent=indent )
        exit()

    count = 0        
    for adir in args.dirs:
        print( adir )
        if args.stats:
            imageStats( adir, indent=indent )
        elif args.py2py3:
            convertDriveToNumpy( adir, indent=indent )
        else:
            data = loadDrive( adir, indent=indent )

            if data is not None:
                if args.desc:
                    output = describeData( data, indent=indent )
                    count += len(data['image_times'])
                    meta = os.path.join( adir, 'meta.txt' )
                    if not os.path.exists(meta):
                        with open(meta, 'w') as f:
                            f.write("Format: MaLPi v1.1\n")
                            f.write(output)
                elif args.extract is not None:
                    extractNewImages( adir, data, args.extract, indent=indent )
                else:
                    if not 'model' in data:
                        print( "{}No model name in drive.pickle. Possibly an older drive format".format( indent ) )
                    else:
                        print( "{}Model: {}".format( indent, data['model'] ) )
                    makeActions( adir, data, indent=indent )
                    isize = (120,120)
                    makeSmallImages( adir, data, size=isize, indent=indent )
                    convertDriveToNumpy( adir, size=isize, indent=indent )
    if count > 0:
        print( "Total samples: {}".format( count ) )
