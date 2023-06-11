#!/usr/bin/env python
# coding: utf-8

# Testing the VAE on the CIFAR10 dataset
import argparse
import os
import sqlite3

import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10
from torchvision import transforms


# Import DonkeyCar, suppressing it's annoying banner
from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    import donkeycar as dk

import pandas as pd
from pathlib import Path
import numpy as np

from malpi.dk.train import preprocessFileList, get_data, get_dataframe, get_dataframe_from_db, get_dataframe_from_db_with_aux
from malpi.dk.vae import VanillaVAE, VAEWithAuxOuts
from malpi.dk.vae_callbacks import TensorboardGenerativeModelImageSampler

from PIL import Image

def show_results( originals, reconstructed, samples ):
    n = len(originals)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(np.transpose(originals[i], (1,2,0)) )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + n+1)
        plt.imshow(np.transpose(reconstructed[i], (1,2,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display samples
        ax = plt.subplot(3, n, i + n + n+1)
        plt.imshow(np.transpose(samples[i], (1,2,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def evaluate( val_dataloader, model, n, device ):

    batch_features, batch_labels = next(iter(val_dataloader))
    #Feature batch shape: torch.Size([32, 3, 32, 32])
    originals = batch_features[:n,:]
    reconstructed = model(originals)
    samples = model.sample(n, device)
    return originals, reconstructed, samples

class AEDataset(torch.utils.data.Dataset):
    """ Convert a dataset intended for categorical output to one that can
        be used to train an autoencoder.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, _ = self.dataset[index]

        return image, image

    def __len__(self):
        return len(self.dataset)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.mnist_train = AEDataset( CIFAR10(self.data_dir, train=True, transform=self.transform) )
            self.mnist_val = AEDataset( CIFAR10(self.data_dir, train=False, transform=self.transform) )
            print( f"Train dataset: {len(self.mnist_train)}" )
            print( f"Val dataset: {len(self.mnist_val)}" )
            #self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=8)

class LitVAE(pl.LightningModule):
    def __init__(self, lr:float=1e-3, image_size: int=128, latent_dim: int=128, beta:float =4.0, notes: str = None):
        super().__init__()
        self.model = VanillaVAE(input_size=image_size, latent_dim=latent_dim, beta=beta)
        #print(self.model)
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_dim = self.model.img_dim

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['image_size'] = (image_size,image_size)
        #self.model.meta['epochs'] = epochs
        self.model.meta['lr'] = lr
        #self.model.meta['batch_size'] = batch_size
        self.model.meta['latent_dim'] = latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def decode(self, z):
        # For use by TensorboardGenerativeModelImageSampler
        return self.model.decode(z)

    def _run_one_batch(self, batch, batch_idx):

        recons, _, mu, log_var = self.model.forward(batch)

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]        

        try:
            loss_vae = self.model.loss_function_exp( batch, recons, mu, log_var )
        except RuntimeError as ex:
            raise

        return recons, loss_vae

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, test_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

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

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    epochs=100
    lr=4.7e-4
    batch_size=128
    z_dim=128
    beta=4.0

    if torch.cuda.is_available():
        #torch.set_default_device('cuda')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cifar10_dm = CIFAR10DataModule()
    model = LitVAE(image_size=32, lr=lr, beta=0.00001)
    img_gen = TensorboardGenerativeModelImageSampler()
    lr_mon = LearningRateMonitor(logging_interval="step"),
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        #logger=CSVLogger(save_dir="logs/"),
        callbacks=[TQDMProgressBar(refresh_rate=10), img_gen],
    )

    trainer.fit(model, cifar10_dm)

    device = torch.device("cpu")
    model.model.to(device)
    o,r,s = evaluate( cifar10_dm.val_dataloader(), model.model, 10, device )
    r = r[0]
    o = o.detach().numpy()
    r = r.detach().numpy()
    s = s.detach().numpy()
    show_results( o, r, s )
