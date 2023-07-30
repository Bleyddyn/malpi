#!/usr/bin/env python
# coding: utf-8

# Train a VAE with or without auxiliary outputs using PyTorch Lightning

import argparse
import os
import sqlite3

import torch
from torchvision import transforms as T
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.tuner import Tuner
from torch.utils.data import random_split, DataLoader

# Import DonkeyCar, suppressing it's annoying banner
from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    import donkeycar as dk

import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image

from malpi.dk.train import preprocessFileList, get_dataframe, get_dataframe_from_db, get_dataframe_from_db_with_aux
from malpi.dk.vae import VanillaVAE, VAEWithAuxOuts
from malpi.dk.vae_callbacks import TensorboardGenerativeModelImageSampler
from malpi.dk.vis import show_vae_results, evaluate_vae, visualize_batch
from malpi.dk.lit import LitVAE, LitVAEWithAux
from malpi.dk.data import ImageAuxDataset, DKImageDataModule

import matplotlib.pyplot as plt

#setup(stage) (str) – either 'fit', 'validate', 'test', or 'predict'
#teardown(stage) (str) – either 'fit', 'validate', 'test', or 'predict'

#on_after_batch_transfer(batch, dataloader_idx)
#on_before_batch_transfer(batch, dataloader_idx)

def get_options():
    # From the fastai version
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
    parser.add_argument('--beta', type=float, default=4.0,
                        help='VAE beta parameter')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize an existing model. Requires --name')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes to be added to the models metadata')
    parser.add_argument('--database', type=str, default=None,
                        help='Path to an sqlite3 database.')
    parser.add_argument('--aux', action='store_true',
                        help='Train with auxiliary data (cte, steering, throttle, track #)')
    return parser.parse_args()

def cli_main():
    cli = LightningCLI(LitVAEWithAux, DKImageDataModule)

""" TODO: Add support for Tensorboard
"""
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

#    cli_main()
#    exit(0)

    model_name = "models/vae_pytorch_v1.pkl"
    input_file = "tubs.sqlite"
    image_size = 128
    epochs=100
    lr=0.0033
    batch_size=128 # Max size according to Tuner is 741
    z_dim=128
    beta=4.0
    notes=None
    aux=False

    alpha_vae = 1.0
    alpha_drive = 0.0
    alpha_track = 0.0

    if torch.cuda.is_available():
        #torch.set_default_device('cuda')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if aux:
        lit = LitVAEWithAux(lr=lr, beta=0.00001)
    else:
        lit = LitVAE(lr=lr, beta=0.00001)
    print( f"KLD_weight: {lit.model.kld_weight}" )
    #lit.hparams.latent_dim = z_dim
    data_model = DKImageDataModule( input_file, batch_size=batch_size, aux=aux)
    if False:
        visualize_batch(data_model)
        exit()

    early_stopping = EarlyStopping('val_loss', patience=5, mode='min', min_delta=0.0)
    # TODO Figure out how to get test images into the Sampler
    img_gen = TensorboardGenerativeModelImageSampler(num_samples=6, nrow=3, data_module=data_model )
    callbacks = None # [early_stopping, img_gen]
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=epochs, logger=False)
    tuner = Tuner(trainer)

    if False:
        tuner.scale_batch_size(lit, datamodule=data_model, mode="binsearch", init_val=128 )

    print( f"lit.hparams: {lit.hparams}" )

    if False:
        lr_finder = tuner.lr_find(lit, datamodule=data_model, min_lr=1e-6, max_lr=1e-2)
        # Results can be found in
        #print(lr_finder.results)

        lr_finder.plot(show=True, suggest=True)

# Pick point based on plot, or get suggestion
        #print( f"suggestion: {lr_finder.suggestion()}" )

        lit.hparams.lr = tuner.suggestion()

    trainer.fit(lit, data_model)

    device = torch.device("cpu")
    lit.model.to(device)
    o,r,s = evaluate_vae( data_model.test_dataloader(), lit.model, 10, device )
    show_vae_results( o, r, s )
