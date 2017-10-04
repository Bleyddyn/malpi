from scipy.misc import imresize
import pickle
with open('drive.pickle','r') as f:
    data = pickle.load(f)

print( "Image count: {}".format( len(data['images']) ) )
print( "Image actions count: {}".format( len(data['image_actions']) ) )
print( "Actions count: {}".format( len(data['actions']) ) )

#act = data['actions']
#print( act )

images = data['images']
small = []
for img in images:
    img = imresize(img, (120,120), interp='nearest' )
    small.append(img)

print( small[0].shape )

with open('images_120x120.pickle','wb') as f:
    pickle.dump( small, f, protocol=pickle.HIGHEST_PROTOCOL )
