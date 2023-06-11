# Visualize data for and the results of a VAE

import os

import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import numpy as np

def show_vae_results( originals, reconstructed=None, samples=None ):
    n = len(originals)

    if isinstance(originals, torch.Tensor):
        originals = originals.detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().numpy()
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().numpy()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(np.transpose(originals[i], (1,2,0)) )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if reconstructed is not None:
            # display reconstruction
            ax = plt.subplot(3, n, i + n+1)
            plt.imshow(np.transpose(reconstructed[i], (1,2,0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if samples is not None:
            # display samples
            ax = plt.subplot(3, n, i + n + n+1)
            plt.imshow(np.transpose(samples[i], (1,2,0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def evaluate_vae( val_dataloader, model, n, device ):

    batch = next(iter(val_dataloader))
    if len(batch) == 2:
        batch_features, batch_labels = batch
    else:
        batch_features = batch

    #Feature batch shape: torch.Size([32, 3, 32, 32])
    originals = batch_features[:n,:]
    print( f"device: {device}" )
    originals = originals.to(device)
    reconstructed = model(originals)
    samples = model.sample(n, device)

    reconstructed = reconstructed[0]

    #originals = originals.detach().numpy()
    #reconstructed = reconstructed.detach().numpy()
    #samples = samples.detach().numpy()

    return originals, reconstructed, samples

def visualize_batch(data_model: pl.LightningDataModule, aux=False):
    data_model.setup()
    train = data_model.train_dataloader()
    batch1 = next(iter(train))
    print( f"Train: {type(batch1)} {len(batch1)} {batch1.shape}" )
    # Get 10 images from the validation set to display
    val_dl = data_model.test_dataloader()
    print( f"test_dl: {type(val_dl)} {len(val_dl)}" )
    batch1 = next(iter(val_dl))
    print( f"Test: {type(batch1)} {len(batch1)} {batch1.shape}" )
    if aux:
        originals = next(iter(data_model.val_dataloader()))[0][:10]
    else:
        originals = next(iter(data_model.val_dataloader()))[:10]
    print( f"originals: {type(originals)} {originals.shape}" )
    show_vae_results( originals )

