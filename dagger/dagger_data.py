import os
import numpy as np
import pickle

def loadOneDrive( drive_dir ):
    actions_file = os.path.join( drive_dir, "image_actions.npy" )
    if os.path.exists(actions_file):
        actions = np.load(actions_file)
        actions = actions.astype('str')
    else:
        actions_file = os.path.join( drive_dir, "image_actions.pickle" )
        with open(actions_file,'r') as f:
            actions = pickle.load(f)

    im_file = os.path.join( drive_dir, "images_120x120.npy" )
    if os.path.exists(im_file):
        images = np.load(im_file)
    else:
        im_file = os.path.join( drive_dir, "images_120x120.pickle" )
        with open(im_file,'r') as f:
            images = pickle.load(f)

    return images, actions

def loadData( dirs, image_norm=True ):
    images = []
    actions = []

    for onedir in dirs:
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir )
            images.extend(dimages)
            actions.extend(dactions)

    images = np.array(images)

    y = actions
    #y = embedActions( actions )
    #y = to_categorical( y, num_classes=5 )
    return images, y

def saveData( path, images, actions ):
    # Ignoring images for now
    actions = np.array(actions)
    actions = actions.astype('str')
    ofname = os.path.join( path, 'image_actions.npy' )
    np.save(ofname, actions)
