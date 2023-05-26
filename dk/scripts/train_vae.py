#!/usr/bin/env python
# coding: utf-8

# Demonstrating how to get DonkeyCar Tub files into a PyTorch/Lightning DataModule
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
    def __init__(self, input_file: str, batch_size=128, num_workers=8, aux=True):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
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
        #test_dataset = self.val_dataset[:20]
        #return torch.utils.data.DataLoader(test_dataset, batch_size=20,
        #            shuffle=False, num_workers=self.num_workers)
        super().test_dataloader()

    def predict_dataloader(self):
        super().predict_dataloader()

    def teardown(self, stage=None):
        super().teardown(stage)

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

class LitVAEWithAux(pl.LightningModule):
    def __init__(self, lr:float=1e-3, image_size: int=128, latent_dim: int=128, beta:float =4.0, notes: str = None):
        super().__init__()
        self.model = VAEWithAuxOuts(input_size=image_size, latent_dim=latent_dim, beta=beta)
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_dim = self.model.img_dim
        self.alpha_vae = 1.0
        self.alpha_drive = 0.0
        self.alpha_track = 0.0

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['image_size'] = (image_size,image_size)
        #self.model.meta['epochs'] = epochs
        self.model.meta['lr'] = lr
        #self.model.meta['batch_size'] = batch_size
        self.model.meta['latent_dim'] = latent_dim
        self.model.meta['loss_alpha'] = (self.alpha_vae, self.alpha_drive, self.alpha_track)
        # From the fastai version.
        #vae.meta['input'] = input_file
        #vae.meta['transforms'] = len(batch_tfms)
        #vae.meta['aux'] = True
        #vae.meta['train'] = len(dls.train_ds)
        #vae.meta['valid'] = len(dls.valid_ds)

        def configure_optimizers(self):
            #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer

        def forward(self, x):
            return self.model(x)

        def decode(self, z):
        # For use by TensorboardGenerativeModelImageSampler
        return self.model.decode(z)

    def _run_one_batch(self, batch, batch_idx):
        images, angle, throttle, cte, track = batch

        # Unsqueeze angle, throttle and cte to make them 2D tensors
        angle = angle.unsqueeze(1)
        throttle = throttle.unsqueeze(1)
        cte = cte.unsqueeze(1)

        recons, mu, log_var, steering_out, throttle_out, cte_out, track_out = self.model.forward(images)

        targets = {"steering": angle, "throttle": throttle, "cte": cte, "track": track}
        targets["images"] = images
        outputs = {"steering": steering_out, "throttle": throttle_out, "cte": cte_out, "track": track_out}
        outputs["images"] = recons

        try:
            loss_vae, loss_drive, loss_track = self.model.loss_function( targets, outputs, mu, log_var )
        except RuntimeError as ex:
            raise

        train_loss = (loss_vae * self.alpha_vae) + (loss_drive * self.alpha_drive) + (loss_track * self.alpha_track)
        return outputs, train_loss

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

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
    USE_CIFAR10=False

    alpha_vae = 1.0
    alpha_drive = 0.0
    alpha_track = 0.0

    if torch.cuda.is_available():
        #torch.set_default_device('cuda')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if USE_CIFAR10:
        cifar10_dm = CIFAR10DataModule()
        model = LitVAE(image_size=32, lr=lr, beta=0.00001)
        img_gen = TensorboardGenerativeModelImageSampler()
        lr_mon = LearningRateMonitor(logging_interval="step"),
        trainer = pl.Trainer(
            max_epochs=30,
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
        exit(0)

    if aux:
        lit = LitVAEWithAux(lr=lr, beta=0.00001)
    else:
        lit = LitVAE(lr=lr, beta=0.00001)
    print( f"KLD_weight: {lit.model.kld_weight}" )
    #lit.hparams.latent_dim = z_dim
    data_model = DKImageDataModule( input_file, batch_size=batch_size, aux=aux)
    if False:
        data_model.setup()
        test = data_model.test_dataloader()
        batch1 = next(iter(test))
        print( f"batch1: {type(batch1)} {len(batch1)} {batch1.shape}" )
        # Get 10 images from the validation set to display
        val_dl = data_model.val_dataloader()
        print( f"val_dl: {type(val_dl)} {len(val_dl)}" )
        batch1 = next(iter(val_dl))
        print( f"batch1: {type(batch1)} {len(batch1)} {batch1.shape}" )
        if aux:
            originals = next(iter(data_model.val_dataloader()))[0][:10]
        else:
            originals = next(iter(data_model.val_dataloader()))[:10]
        print( f"originals: {type(originals)} {originals.shape}" )
        show_results( originals, originals, originals )
        exit()

    early_stopping = EarlyStopping('val_loss', patience=5, mode='min', min_delta=0.0)
    # TODO Figure out how to get test images into the Sampler
    img_gen = TensorboardGenerativeModelImageSampler(num_samples=6, nrow=3 )
    #trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer = pl.Trainer(callbacks=[early_stopping, img_gen], max_epochs=epochs)
    trainer.fit(lit, data_model)
    exit()

    #dls = get_data(None, df_all=df, item_tfms=item_tfms, batch_tfms=batch_tfms, verbose=False, autoencoder=True, aux=aux)
    #vae.meta['train'] = len(dls.train_ds)
    #vae.meta['valid'] = len(dls.valid_ds)

    optimizer = torch.optim.SGD(vae.parameters(), lr=lr)

    if False:
        print( "Preloading all data" )
        for idx, (images, angle, throttle, cte, track) in enumerate(dataloader_train):
            pass
        print( "   Finished" )

    epochs = 3
    for epoch in range(epochs):

        vae = vae.train()
        t_loss_list, v_loss_list = [], []
        for idx, (images, angle, throttle, cte, track) in enumerate(dataloader_train):
            # Unsqueeze angle, throttle and cte to make them 2D tensors
            angle = angle.unsqueeze(1)
            throttle = throttle.unsqueeze(1)
            cte = cte.unsqueeze(1)

            recons, mu, log_var, steering_out, throttle_out, cte_out, track_out = vae.forward(images)

            targets = {"steering": angle, "throttle": throttle, "cte": cte, "track": track}
            targets["images"] = images
            outputs = {"steering": steering_out, "throttle": throttle_out, "cte": cte_out, "track": track_out}
            outputs["images"] = recons

            try:
                loss_vae, loss_drive, loss_track = vae.loss_function( targets, outputs, mu, log_var )
            except RuntimeError as ex:
                raise

            train_loss = (loss_vae * alpha_vae) + (loss_drive * alpha_drive) + (loss_track * alpha_track)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(
                    f"Epoch {epoch+1:02d}/{epochs:02d}"
                    f" | Batch {idx:02d}/{len(dataloader_train):02d}"
                    f" | Train Loss {train_loss:.3f} ({loss_vae:.3f} {loss_drive:.3f} {loss_track:.3f})"
                )

        # Validation


