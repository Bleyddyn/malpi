""" A script that trains a model on each input file.
	Input: a list of files (each of which has a list of tubs), an output directory
	Output: one model for each input file (named e.g. 20220819_1010_track#.pkl)
	TODO: Log (to a file in the output dir): Date/time, List of input tubs, total number of training samples
	Run the test script as part of this one?

Track names returned by the Simulator:
generated_road
warehouse
sparkfun_avc
generated_track
roboracingleague_1
waveshare
mini_monaco
warren
circuit_launch
mountain_track

Track names as defined in gym_donkeycar:
donkey-generated-roads-v0
donkey-warehouse-v0
donkey-avc-sparkfun-v0
donkey-generated-track-v0
donkey-roboracingleague-track-v0
donkey-waveshare-v0
donkey-minimonaco-track-v0
donkey-warren-track-v0
donkey-circuit-launch-track-v0
donkey-thunderhill-track-v0
all
"""

import os
import argparse
import sqlite3
from datetime import datetime
import json

import torch
from torch import nn
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

from malpi.dk.train import get_dataframe_from_db_with_aux, get_track_metadata
from malpi.dk.test import main as gym_test, print_results
from malpi.dk.lit import LitVAE, LitVAEWithAux, DKDriverModule, DKRNNDriverModule
from malpi.dk.data import ImageZDataset, DKImageZDataModule, DKImageZSequenceDataModule

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description='Train a model on each input file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--vae_model', type=str, default=None,help='File with a pre-trained VAE model. If not specified, train a non VAE model.')
    parser.add_argument('--database', type=str, default=None,help='Path to an sqlite3 database.')
    parser.add_argument('--one_model', action='store_true', default=False, help='Train a single model on all the input files.')
    parser.add_argument('--rnn', action='store_true', default=False, help='Train an RNN model.')
    parser.add_argument('--no_logging', action='store_true', default=False, help='Disable logging.')
    parser.add_argument('--notes', type=str, default=None, help='Notes to be added to the models metadata')
    parser.add_argument('output_dir', help='Output directory')
    args = parser.parse_args()

    vae_model_path = args.vae_model
    epochs = args.epochs
    lr = args.lr
    batch_size = 5
    early_stopping = EarlyStopping('val_loss', patience=5, mode='min', min_delta=0.0)
    callbacks = [early_stopping]

    results = None

    vae_model = LitVAE.load_from_checkpoint(vae_model_path)
    vae_model.eval()

    # Always pull auxilliary data from the database, but only use it if self.aux is True
    with sqlite3.connect(args.database) as conn:
        df_all = get_dataframe_from_db_with_aux( input_file=None, conn=conn, sources=None )

    if args.one_model:
        if args.rnn:
            data_model = DKImageZSequenceDataModule(vae_model, df_all, batch_size=batch_size, sequence_length=100, shuffle=False)
            lit = DKRNNDriverModule(batch_size=batch_size, latent_dim=vae_model.latent_dim, hidden_size=100, notes=args.notes )
        else:
            data_model = DKImageZDataModule(vae_model, df_all, batch_size=256, shuffle=True)
            lit = DKDriverModule(notes=args.notes)
        trainer = pl.Trainer(callbacks=callbacks, max_epochs=epochs, logger=(not args.no_logging))
        trainer.fit(lit, data_model)

        # Test the model on all tracks
        # This may fail if the simulator isn't running
        try:
            model_path = trainer.checkpoint_callback.best_model_path
            lit.to('cpu')
            vae_model.to('cpu')
            results = gym_test('all', lit, model_path, vae_model)
        except Exception as e:
            print( f"Exception {e} while testing model {model_path}" )

    else:
        with sqlite3.connect(args.database) as conn:
            track_meta = get_track_metadata(conn)
        track_ids = df_all["track_id"].unique()
        results = []
        for track in track_ids:
            track_name = track_meta[track][0]
            data_model = DKImageZDataModule(vae_model, df_all, track_id=track, batch_size=256)
            if args.rnn:
                lit = DKRNNDriverModule(latent_dim=vae_model.latent_dim, hidden_size=100, notes=args.notes )
            else:
                lit = DKDriverModule(notes=args.notes)
            trainer = pl.Trainer(callbacks=callbacks, max_epochs=epochs, logger=(not args.no_logging))
            trainer.fit(lit, data_model)

            # Test the model
            try:
                model_path = trainer.checkpoint_callback.best_model_path
                lit.to('cpu')
                vae_model.to('cpu')
                res = gym_test(track_name, lit, model_path, vae_model=vae_model)
                results.append( res )
                vae_model.to('gpu')
            except Exception as e:
                print( f"Exception {e} while testing model {model_path}" )

    if results is not None:
        # Export the results to json
        with open(args.output_dir + '/results.json', 'w') as f:
            json.dump(results, f)
        if isinstance(results, dict):
            print_results(results)
        elif isinstance(results, list):
            for r in results:
                print_results(r)
