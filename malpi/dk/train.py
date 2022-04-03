import os
from pathlib import Path

import torch

# TODO Only import what we need
#from fastai.data.all import *
from fastai.data.block import DataBlock, RegressionBlock
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize
#from fastai.vision.all import *
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.layers import ConvLayer, Flatten
from fastai.learner import Learner
from fastai.callback.hook import ActivationStats
from fastai.losses import MSELossFlat
from fastai.metrics import rmse

from donkeycar.parts.tub_v2 import Tub
import pandas as pd

from malpi.dk.autoencoder import Autoencoder

def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessFileList( filelist ):
    """ Given a list of filenames, open each one and read each line.
        Put each line into a list of directories.
        Remove comments (line starts with #) and blank lines.
        Return the list of directories.
    """
    dirs = []
    if filelist is not None:
        for afile in filelist:
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    removeComments( dirs )
    return dirs

def tubs_from_filelist(file_list, verbose=False):
    """ Load all tubs listed in all files in file_list """
    tub_dirs = preprocessFileList(file_list)
    tubs = []
    count = 0
    root_path = Path("data")
    for item in tub_dirs:
        if Path(item).is_dir():
            try:
                t = Tub(str(item),read_only=True)
            except FileNotFoundError as ex:
                continue
            except ValueError as ex:
                # In case the catalog file is empty
                continue
            tubs.append(t)
            count += len(t)
    if verbose:
        print( f"Loaded {count} records." )

    return tubs

def tubs_from_directory(tub_dir, verbose=False):
    """ Load all tubs in the given directory """
    tubs = []
    count = 0
    root_path = Path(tub_dir)
    for item in root_path.iterdir():
        if item.is_dir():
            try:
                t = Tub(str(item),read_only=True)
                count += len(t)
            except FileNotFoundError as ex:
                continue
            except ValueError as ex:
                # In case the catalog file is empty
                continue
            tubs.append(t)
    if verbose:
        print( f"Loaded {count} records." )

    return tubs

def dataframe_from_tubs(tubs):
    dfs = []
    for tub in tubs:
        df = pd.DataFrame(tub)
        name = Path(tub.base_path).name
        pref = os.path.join(tub.base_path, Tub.images() ) + "/"
        df["cam/image_array"] = pref + df["cam/image_array"]
        dfs.append(df)
        #print( f"Tub {name}: {df['user/throttle'].min()} - {df['user/throttle'].max()}" )
    return pd.concat(dfs)

def get_dataframe(inputs, verbose=False):
    tubs = None

    try:
        input_path = Path(inputs)
        if input_path.is_dir():
            tubs = tubs_from_directory(input_path)
    except TypeError as ex:
        pass

    if tubs is None:
        if isinstance(inputs, str):
            inputs = [inputs]
        tubs = tubs_from_filelist(inputs)

    if tubs is None:
        if verbose:
            print( f"No tubs found at {inputs}")
        return None

    df_all = dataframe_from_tubs(tubs)

    if verbose:
        df_all.describe()

    return df_all

def get_data(inputs, df_all=None, batch_tfms=None, item_tfms=None, verbose=False, autoencoder=False):

    if df_all is None:
        df_all = get_dataframe(inputs, verbose)

    if item_tfms is None:
        tfms = [Resize(128,method="squish")]
    else:
        tfms = item_tfms

    if autoencoder:
        blocks = (ImageBlock, ImageBlock)
        y_reader = ColReader("cam/image_array")
    else:
        blocks = (ImageBlock, RegressionBlock(n_out=2))
        y_reader = ColReader(['user/angle','user/throttle'])

    pascal = DataBlock(blocks=blocks,
                       splitter=RandomSplitter(),
                       get_x=ColReader("cam/image_array"),
                       get_y=y_reader,
                       item_tfms=tfms,
                       batch_tfms=batch_tfms,
                       n_inp=1)

    dls = pascal.dataloaders(df_all)

    if verbose:
        dls.show_batch()
        dls.one_batch()[0].shape

    return dls

def get_learner(dls):
    model = torch.nn.Sequential(
        ConvLayer(3, 24, stride=2),
        ConvLayer(24, 32, stride=2),
        ConvLayer(32, 64, stride=2),
        ConvLayer(64, 128, stride=2),
        ConvLayer(128, 256, stride=2),
        torch.nn.AdaptiveAvgPool2d(1),
        Flatten(),
        torch.nn.Linear(256, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, dls.c),
        torch.nn.Tanh()
        )
#print(model)
    callbacks=ActivationStats(with_hist=True)
    learn = Learner(dls, model,  loss_func = MSELossFlat(), metrics=[rmse], cbs=callbacks)
    #valley = learn.lr_find()
    return learn

def get_autoencoder(dls, verbose=True):
    ae = Autoencoder( z_size=64, input_dimension=(128,128,3) )
    model = torch.nn.Sequential( ae )
    if verbose:
        print(model)
    callbacks=ActivationStats(with_hist=True)
    #learn = Learner(dls, model,  loss_func = nn.BCELoss(), metrics=[rmse], cbs=callbacks)
    learn = Learner(dls, model,  loss_func = torch.nn.MSELoss(), metrics=[rmse], cbs=callbacks)
    return learn

def train_autoencoder( input_file, epochs, lr, name, verbose=True ):
    item_tfms = [Resize(128,method="squish")]
    dls = get_data(input_file, item_tfms=item_tfms, verbose=verbose, autoencoder=True)
    learn = get_autoencoder(dls, verbose=verbose)
    learn.fit_one_cycle(epochs, lr)
    learn.export( name + ".pkl" )
    #learn.recorder.plot_loss()
    #learn.show_results(figsize=(20,10))
    #plt.savefig(name + '.png')
    return learn, dls
