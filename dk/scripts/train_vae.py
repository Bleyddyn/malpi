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

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ImageAuxDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, aux=True):
        self.dataframe = dataframe
        self.aux = aux
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Resize(128, antialias=True) # TODO Get this from the loaded model
            #T.Normalize( mean=0.5, std=0.2 )
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.dataframe.iloc[index]
        img_path = row['cam/image_array']
        img = Image.open(img_path)
        img = self.preprocess(img)

        if self.aux:
            return (
                img,
                row['user/angle'],
                row['user/throttle'],
                row['pos_cte'],
                row['track_id'].astype(np.int64)
            )
        else:
            return img

class DKImageDataModule(pl.LightningDataModule):
    def __init__(self, input_file: str, batch_size=128, num_workers=8, aux=True, test_batch_size=20):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.aux = aux
        self.df_all = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        with sqlite3.connect(self.input_file) as conn:
            self.df_all = get_dataframe_from_db_with_aux( input_file=None, conn=conn, sources=None )
        dataset = ImageAuxDataset(self.df_all, self.aux)
# See: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,[0.8,0.2])
        #print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Select a random subset of the validation dataset
        subset_indices = torch.randint(0, len(self.val_dataset), (self.test_batch_size,) )
        subset = torch.utils.data.Subset(self.val_dataset, subset_indices)
        return torch.utils.data.DataLoader(subset, batch_size=self.test_batch_size,
                    shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        super().predict_dataloader()

    def teardown(self, stage=None):
        super().teardown(stage)

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
    lr=4.7e-4
    batch_size=128
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
    trainer = pl.Trainer(callbacks=[early_stopping, img_gen], max_epochs=epochs)
    trainer.fit(lit, data_model)

    device = torch.device("cpu")
    lit.model.to(device)
    o,r,s = evaluate_vae( data_model.test_dataloader(), lit.model, 10, device )
    show_vae_results( o, r, s )
