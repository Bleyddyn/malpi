import os
import pickle
import numpy as np

import model_keras
from keras.utils import to_categorical

def describeDriveData( data ):
    print( data.keys() )
    for key, value in data.iteritems():
        try:
            print( "{} length {}".format( key, len(value) ) )
        except:
            pass

def embedActions( actions ):
    embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
    emb = []
    for act in actions:
        emb.append( embedding[act] )
    return emb

drive_dir = "/Users/andrew/Dev/RaspberryPI/drive/drive_20170930_124550"
drive_file = os.path.join( drive_dir, "drive.pickle" )

with open(drive_file,'r') as f:
    data = pickle.load(f)
    #data = pickle.load(f fix_imports=True, encoding='bytes')

images = np.array(data['images'])
actions = data['image_actions']
y = embedActions( actions )
y = to_categorical( y, num_classes=5 )

input_dim = images[0].shape
num_actions = len(y[0])
model = model_keras.make_model_orig( num_actions, input_dim )

model.fit( images, y )
