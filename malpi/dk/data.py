""" Dataset and LightningDataModules for training VAE's and drivers. """

import os

import torch
from torchvision import transforms as T
import lightning.pytorch as pl
from torch.utils.data import random_split

# Import DonkeyCar, suppressing it's annoying banner
from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    import donkeycar as dk

import numpy as np
from PIL import Image
import sqlite3

from malpi.dk.train import get_dataframe_from_db_with_aux

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
    def __init__(self, input_file: str, batch_size=128, num_workers=8, aux=True, test_batch_size=20, shuffle=True):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.aux = aux
        self.shuffle = shuffle
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
                    shuffle=self.shuffle, num_workers=self.num_workers)

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
    def __init__(self, vae_model, dataframe, batch_size=128, num_workers=8, aux=False,
            track_id: int = None, test_batch_size=20, sampler=None, shuffle=True):
        super().__init__()
        self.vae_model = vae_model
        self.df_all = dataframe
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.aux = aux
        self.track_id = track_id
        self.sampler = sampler
        self.shuffle = True if sampler is not None else shuffle
        self.train_dataset = None
        self.val_dataset = None
        self.verbose = False

    def add_z( self, df_view ):
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
        for i in range(0, len(df_view), self.batch_size):
            if self.verbose:
                print(f'From {i} to {i+self.batch_size} of {len(df_view)}')
            batch = df_view.iloc[i:i+self.batch_size]
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

        df_view['mu'] = mu_series
        df_view['log_var'] = logvar_series
        return df_view

    def filter_for_track( self, track_id ):
        """ Filter the dataframe for rows with the given track name """
        return self.df_all.loc[self.df_all['track_id'] == track_id]

    def setup(self, stage=None):
        if self.track_id is not None:
            df_view = self.filter_for_track( self.track_id ).copy()
        else:
            df_view = self.df_all
        df_view = self.add_z(df_view)
        dataset = ImageZDataset(df_view, self.aux)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,[0.8,0.2])

    def train_dataloader(self):
        if self.sampler is not None:
            sampler = self.sampler(self.train_dataset)
        else:
            sampler = None
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                    shuffle=self.shuffle, num_workers=self.num_workers, sampler=sampler)

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

class ImageZSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, seq_len, aux=True):
        self.dataframe = dataframe
        self.seq_len = seq_len
        self.aux = aux
        self.indexes = []
        self.find_sequences()

    def find_sequences(self):
        """ Find starting index of all sequences in the dataframe.
            TODO: Make sure no sequence crosses too large of a time gap.
                Vary exact starting index if there's spare room. Different each epoch?.
        """
        ignore_short = len(self.dataframe)
        ignore_short -= ignore_short % self.seq_len
        self.indexes = range(0, ignore_short, self.seq_len)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        start = self.indexes[index]
        row = self.dataframe.iloc[start:(start+self.seq_len)]

        mu_all = []
        var_all = []
        for i in range(start, (start+self.seq_len)):
            mu = row['mu'][i]
            mu_all.append( mu )
            log_var = row['log_var'][i]
            var_all.append(log_var)
        mu = torch.Tensor(np.array(mu_all))
        log_var = torch.Tensor(np.array(var_all))
        steer = torch.Tensor(row['user/angle'].to_numpy())
        throttle = torch.Tensor(row['user/throttle'].to_numpy())

        return (mu, log_var, steer, throttle)

        if self.aux:
            ret += (row['pos_cte'], row['track_id'].astype(np.int64))

        return ret


class DKImageZSequenceDataModule(pl.LightningDataModule):
    def __init__(self, vae_model, dataframe, batch_size=128, sequence_length=100, num_workers=8, aux=False,
            track_id: int = None, test_batch_size=20, sampler=None, shuffle=True):
        super().__init__()
        self.vae_model = vae_model
        self.df_all = dataframe
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.aux = aux
        self.track_id = track_id
        self.sampler = sampler
        self.shuffle = True if sampler is not None else shuffle
        self.train_dataset = None
        self.val_dataset = None
        self.verbose = False

    def add_z( self, df_view ):
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
        for i in range(0, len(df_view), self.batch_size):
            if self.verbose:
                print(f'From {i} to {i+self.batch_size} of {len(df_view)}')
            batch = df_view.iloc[i:i+self.batch_size]
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

        df_view['mu'] = mu_series
        df_view['log_var'] = logvar_series
        return df_view

    def filter_for_track( self, track_id ):
        """ Filter the dataframe for rows with the given track name """
        return self.df_all.loc[self.df_all['track_id'] == track_id]

    def setup(self, stage=None):
        if self.track_id is not None:
            df_view = self.filter_for_track( self.track_id ).copy()
        else:
            df_view = self.df_all
        df_view = self.add_z(df_view)
        dataset = ImageZSequenceDataset(df_view, self.sequence_length, self.aux)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,[0.8,0.2])

    def train_dataloader(self):
        if self.sampler is not None:
            sampler = self.sampler(self.train_dataset)
        else:
            sampler = None
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                    shuffle=self.shuffle, num_workers=self.num_workers, sampler=sampler)

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

