"""
Reconstruct existing and generate random images using a VAE. Latent models are not yet supported.

Usage:
    sample.py [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] [--type=(latent|vae)] [--count=<count>] [--z_dim=<zsize>] [--meta] <model>

Options:
    -h --help           Show this screen.
    -m --meta           Display meta information for the model, then exit.
    -f --file=<file>    A text file containing paths to tub files, one per line. Option may be used more than once.
    -c --count=<count>  How many samples to reconstruct/generate. [default: 10]
    -z --z_dim=<zsize>  How many dimensions in the latent space. [default: 512]
"""

import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from docopt import docopt
import donkeycar as dk
from donkeycar.utils import get_model_by_type, gather_records, load_scaled_image_arr
from donkeycar.templates.train import collate_records
from vae_model import KerasVAE

def sample_vae(vae, dirs, count):
    z_size = vae.z_dim
    batch_size=count

    z = np.random.normal(size=(batch_size,z_size))
    samples = vae.decode(z)
    input_dim = samples.shape[1:]

    n = batch_size
    plt.figure(figsize=(20, 6), tight_layout=False)
    plt.title('VAE samples')
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Random")

    data = {}
    orig = []
    opts = { 'cfg' : cfg, 'categorical': False}
    records = gather_records(cfg, dirs, opts, verbose=True)
    collate_records(records, data, opts)
    keys = list(data.keys())
    keys = shuffle(keys)[0:batch_size]
    for key in keys:
        filename = data[key]['image_path']
        img_arr = load_scaled_image_arr(filename, cfg)
        orig.append(img_arr)

    orig = np.array(orig)
    recon = vae.decode( vae.encode(orig) )

    for i in range(n):
        ax = plt.subplot(3, n, n+i+1)
        plt.imshow(orig[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Real")

        ax = plt.subplot(3, n, (2*n)+i+1)
        plt.imshow(recon[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Reconstructed")

    plt.savefig( "samples_vae.png" )
    plt.show()


def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessFileList( filelist ):
    dirs = []
    if filelist is not None:
        for afile in filelist:
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    removeComments( dirs )
    return dirs

def model_meta( model ):
    """ Check a model to see if has auxiliary outputs, if so return the number of outputs. """
    aux = None
    z_dim = None
    dropout = None
    fname = model[:-10] + "model.json"
    try:
        with open(fname,'r') as f:
            json_str = f.read()
            data = json.loads(json_str)
            layers = data["config"]["layers"]
            for l in layers:
                if l.get("name","") == "aux_output":
                    aux = l["config"]["units"]
                elif l.get("name","") == "z_mean":
                    z_dim = l["config"]["units"]
                elif l.get("name","").startswith("SpatialDropout_"):
                    # e.g. SpatialDropout_0.4_1
                    dropout = float( l.get("name","").split("_")[1])
    except:
        pass

    return z_dim, dropout, aux


def main(model, model_type, dirs, count, cfg, z_dim, aux=0, dropout=None):
    #kl = get_model_by_type(model_type, cfg=cfg)
    input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)
    kl = KerasVAE(input_shape=input_shape, z_dim=z_dim, aux=aux, dropout=dropout)
    kl.set_weights(model, by_name=True)
    sample_vae(kl, dirs, count)

if __name__ == "__main__":
    args = docopt(__doc__)

    if args['--meta']:
        z, dropout, aux = model_meta(args['<model>'])
        print( "Z_dim: {}".format( z ) )
        print( "  Aux: {}".format( aux ) )
        print( " Drop: {}".format( dropout ) )
        exit()

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")

    tub = args['--tub']
    model = args['<model>']
    model_type = args['--type']
    count = int(args['--count'])
    z_dim = int(args['--z_dim'])
    z, dropout, aux = model_meta(model)
    if z is not None:
        z_dim = z
    if aux is None:
        aux = 0

    print( "Model meta: {} auxiliary outputs, z_dim {}, dropout {}".format( aux, z_dim, dropout ) )

    dirs = preprocessFileList( args['--file'] )
    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend( tub_paths )

    main(model, model_type, dirs, count, cfg, z_dim, aux=aux, dropout=dropout)
