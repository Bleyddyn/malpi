#!/usr/bin/env python
# coding: utf-8

# Demonstrating how to get DonkeyCar Tub files into a PyTorch/fastai DataBlock
import argparse
import os

import matplotlib.pyplot as plt

from fastai.data.all import *
from fastai.vision.all import *
from fastai.data.transforms import ColReader, Normalize, RandomSplitter

import torch
from torchvision import transforms as T

from donkeycar.parts.tub_v2 import Tub
import pandas as pd
from pathlib import Path

from malpi.dk.train import preprocessFileList, get_data
from malpi.dk.vae import VanillaVAE

from PIL import Image

def get_images( sample_dir, image_size=128 ):
    images = []
    # load all images in sample_dir using PIL, resize and convert to numpy array
    for filename in os.listdir(sample_dir):
        try:
            img = Image.open( os.path.join(sample_dir, filename) )
            img = img.resize((image_size, image_size))
            img = np.array(img)
            img = np.transpose(img,(2,0,1))
            img = img.astype(np.float32) / 255.0
            images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            #pass
    images = np.asarray(images)
    return images

def show_results( model_path, sample_dir ):
    preprocess = T.Compose([
        T.Resize(128), # TODO Get this from the loaded model
        T.ToTensor(),
        T.Normalize( mean=0.5, std=0.2 )
    ])

    learner = load_learner(model_path)
    # Create a fastai dataset from sample_dir, then use the learner to predict
    #dls = ImageDataLoaders.from_folder(sample_dir, item_tfms=Resize(128,method="squish"))
    images = get_images( sample_dir )
    img_tensor = torch.from_numpy(images)
    #img_tensor = torch.unsqueeze( img_tensor, 0 )
    reconstructed, in_tensor, mu, log_var = learner.model.forward(img_tensor)
    reconstructed = reconstructed.detach().numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(np.transpose(images[i], (1,2,0)) )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(np.transpose(reconstructed[i], (1,2,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def train_vae( input_file, model_name, epochs=100, lr=4.7e-4, z_dim=128, notes=None ):

    random_resize = False

    if model_name is None:
        # while model_name exists, increment the version number
        model_name = "models/vae_v1.pkl"
        i = 1
        while os.path.exists(model_name):
            i += 1
            model_name = f"models/vae_v{i}.pkl"

    image_size = 128

    if random_resize:
        item_tfms = [RandomResizedCrop(image_size,p=0.5,min_scale=0.5,ratio=(0.9,1.1))]
    else:
        item_tfms = [Resize(image_size,method="squish")]
    batch_tfms = [RandomErasing(sh=0.1, max_count=6,p=0.5),
                  Brightness(max_lighting=0.4),
                  Contrast(max_lighting=0.4),
                  Saturation(max_lighting=0.4)]
    callbacks = [EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=5)]

    vae = VanillaVAE(input_size=image_size, latent_dim=z_dim)

    vae.meta['input'] = input_file
    vae.meta['image_size'] = (image_size,image_size)
    vae.meta['epochs'] = epochs
    vae.meta['lr'] = lr
    vae.meta['latent_dim'] = z_dim
    vae.meta['random_resize'] = random_resize
    vae.meta['transforms'] = len(batch_tfms)
    if notes is not None:
        vae.meta['notes'] = notes

    dls = get_data(input_file, item_tfms=item_tfms, batch_tfms=batch_tfms, verbose=False, autoencoder=True)
    vae.meta['train'] = len(dls.train_ds)
    vae.meta['valid'] = len(dls.valid_ds)

    learn = Learner(dls, vae, loss_func=vae.loss_function)
    learn.fit_one_cycle(epochs, lr, cbs=callbacks)
    if not model_name.endswith(".pkl"):
        model_name += ".pkl"
    learn.export( model_name )

def get_options():
    parser = argparse.ArgumentParser(description='Train a VAE on DonkeyCar Data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default="tracks_all.txt",
                        help='Input file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=4.7e-4,
                        help='Learning rate')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the VAE model. Default will be models/vae_v#.pkl (with # being the next available number)')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize an existing model. Requires --name')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes to be added to the models metadata')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_options()

    if args.vis:
        show_results(args.name, sample_dir="vae_test_images")
    else:
        train_vae( input_file=args.input, model_name=args.name, epochs=args.epochs, lr=args.lr, z_dim=args.z_dim, notes=args.notes )
