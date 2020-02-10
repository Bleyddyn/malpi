import os
import numpy as np

def load_tub_npz( dirs, base_dir="", max_images=None, verbose=False ):
    """ Load images from pre-processed tub.npz files.
        dirs should be a list of tub names
        load each: base_dir + tub_name + '.npz' (extension is added if needed)
        Add all images from npz['images']
        
    """
    images = []
    total = 0
    for idx, tub in enumerate(dirs):
        if verbose:
            print( "Loading {}".format( tub ) )
        if not tub.endswith('.npz'):
            tub += '.npz'
        data = np.load( os.path.join( base_dir, tub ) )
        if verbose:
            print( "  {}".format( data["images"].shape ) )
        images.append( data['images'] )
        total += data['images'].shape[0]
        if max_images is not None and total > max_images:
            break

    images = np.concatenate( images, axis=0 ).astype(np.float32) / 255.0

    if verbose:
        print( "Loaded {} images {}".format( images.shape[0], images.shape[1:] ) )

    return images
