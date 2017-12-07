import os
import pickle
from scipy.misc import imresize

def dumpArrayToFile( data, key ):
    fname = key + ".pickle"
    array = data[key]
    with open(fname, 'w') as f:
        pickle.dump( array, f, protocol=pickle.HIGHEST_PROTOCOL )

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

if not os.path.exists('image_actions.pickle'):
    with open('drive.pickle','r') as f:
        data = pickle.load(f)
    print( "{} Image actions".format( len(data['image_actions'] ) ) )
    dumpArrayToFile( data, 'image_actions' )
else:
    print( "Image actions already exists" )

if not os.path.exists('images_120x120.pickle'):
    smallImages = []
    for fname in os.listdir("."):
        if fname.startswith("images_") and fname.endswith(".pickle"):
            print( "Reading {}".format(fname) )
            im1 = getImagesFrom(fname)
            smallImages.extend(im1)

    print( "writing {} small images".format( len(smallImages) ) )

    with open('images_120x120.pickle','wb') as f:
        pickle.dump( smallImages, f, protocol=pickle.HIGHEST_PROTOCOL )
else:
    print( "Small Images file already exists" )

#with open('drive.pickle','r') as f:
#    data = pickle.load(f)

#with open('accel.pkl','r') as f:
#    data = pickle.load(f)

#with open('drive3.pickle', 'wb') as f3:
#    pickle.dump( data, f3, protocol=pickle.HIGHEST_PROTOCOL )

#data3 = { }
#
#print( data.keys() )
#for key in data.keys():
#    print( key )
#
#data = data3
#print( "Image count: {}".format( len(data['images']) ) )
#print( "Image actions count: {}".format( len(data['image_actions']) ) )
#print( "Actions count: {}".format( len(data['actions']) ) )
#
#act = data['actions']
#print( act )

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

#writeToSeparateFiles()
#readFromSeparateFiles()
