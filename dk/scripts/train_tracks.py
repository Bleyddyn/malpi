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

from malpi.dk.vae import SplitDriver
from malpi.dk.train import get_dataframe_from_db_with_aux, get_track_metadata
from malpi.dk.test import main as gym_test, print_results
from malpi.dk.lit import LitVAE, LitVAEWithAux, DKDriverModule

class ImageZDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, aux=True):
        self.dataframe = dataframe
        self.aux = aux

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.dataframe.iloc[index]
        mu = row['mu']
        log_var = row['log_var']

        ret = (mu, log_var, row['user/angle'], row['user/throttle'])

        if self.aux:
            ret += (row['pos_cte'], row['track_id'].astype(np.int64))

        return ret

class DKImageZDataModule(pl.LightningDataModule):
    def __init__(self, db_file: str, vae_model, dataframe=None, batch_size=128, num_workers=8, aux=False,
            track_id: int = None, test_batch_size=20):
        super().__init__()
        self.db_file = db_file
        self.vae_model = vae_model
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.aux = aux
        self.track_id = track_id
        self.df_all = dataframe
        self.train_dataset = None
        self.val_dataset = None
        self.verbose = False

    def add_z( self ):
        """ Calculate mu and logvar values from the images using the vae_model
            and add them to the dataframe """

        preprocess = T.Compose([
                T.ToTensor(),
                T.Resize(128, antialias=True)
            ])

# for each batch of rows from the dataframe, stack the images and run them through the VAE
# then add the mu and logvar values to the dataframe
        mu_series = []
        logvar_series = []
        for i in range(0, len(self.df_all), self.batch_size):
            if self.verbose:
                print(f'From {i} to {i+self.batch_size} of {len(self.df_all)}')
            batch = self.df_all.iloc[i:i+self.batch_size]
            images = []
            for index, row in batch.iterrows():
                img_path = row['cam/image_array']
                img = Image.open(img_path)
                img = preprocess(img)
                images.append(img)
            images = torch.stack(images).to('cuda')
            _, _, mu, logvar = self.vae_model(images)
            mu = mu.detach().cpu().numpy()
            logvar = logvar.detach().cpu().numpy()
            mu_series.extend(mu)
            logvar_series.extend(logvar)

        self.df_all['mu'] = mu_series
        self.df_all['log_var'] = logvar_series

    def filter_for_track( self, track_id ):
        """ Filter the dataframe for rows with the given track name """
        return self.df_all.loc[self.df_all['track_id'] == track_id]

    def setup(self, stage=None):
        if self.df_all is None:
            # Always pull auxilliary data from the database, but only use it if self.aux is True
            with sqlite3.connect(self.db_file) as conn:
                self.df_all = get_dataframe_from_db_with_aux( input_file=None, conn=conn, sources=None )
            self.add_z()
        if self.track_id is not None:
            df_view = self.filter_for_track( self.track_id )
        else:
            df_view = self.df_all
        dataset = ImageZDataset(df_view, self.aux)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,[0.8,0.2])

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

def get_track_name( filename ):
    """ Open the file and check each line that starts with a '#',
        possibly followed by whitespace,
        followed by the string "Track: ",
        followed by the track name.
    """
    with open(filename) as f:
        for line in f:
            if line.startswith('#') and 'Track: ' in line:
                return line.split(':')[1].strip()
    return None

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description='Train a model on each input file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--vae_model', type=str, default=None,help='File with a pre-trained VAE model. If not specified, train a non VAE model.')
    parser.add_argument('--database', type=str, default=None,help='Path to an sqlite3 database.')
    parser.add_argument('--one_model', action='store_true', default=False, help='Train a single model on all the input files.')
    parser.add_argument('--no_logging', action='store_true', default=False, help='Disable logging.')
    parser.add_argument('--notes', type=str, default=None, help='Notes to be added to the models metadata')
    parser.add_argument('output_dir', help='Output directory')
    args = parser.parse_args()

    vae_model_path = args.vae_model # "models/vae_v3.pkl"
    epochs = args.epochs
    lr = args.lr
    early_stopping = EarlyStopping('val_loss', patience=5, mode='min', min_delta=0.0)
    callbacks = [early_stopping]

    results = None

    vae_model = LitVAE.load_from_checkpoint(vae_model_path)
    vae_model.eval()
    data_model = DKImageZDataModule("tubs.sqlite", vae_model, batch_size=256)

    if args.one_model:
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
        data_model.setup()
        df_all = data_model.df_all
        track_ids = df_all["track_id"].unique()
        results = []
        for track in track_ids:
            track_name = track_meta[track][0]
            data_model = DKImageZDataModule("tubs.sqlite", vae_model, dataframe=df_all, track_id=track, batch_size=256)
            data_model.setup()

            print( f"Training on track {track} with {len(data_model.train_dataloader())} batches" )
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
