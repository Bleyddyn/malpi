import os
from pathlib import Path
import sqlite3

import torch

# TODO Only import what we need
#from fastai.data.all import *
from fastai.data.block import DataBlock, RegressionBlock, CategoryBlock
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize
#from fastai.vision.all import *
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.layers import ConvLayer, Flatten
from fastai.learner import Learner, load_learner
from fastai.callback.hook import ActivationStats
from fastai.losses import MSELossFlat
from fastai.metrics import rmse

from donkeycar.parts.tub_v2 import Tub
import pandas as pd
import numpy as np

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

        # if user/angle is 0 and pilot/angle exists, use pilot/angle instead. Same for throttle.
        if 'pilot/angle' in df.columns:
            df.loc[df['user/angle'] == 0, 'user/angle'] = df['pilot/angle'].fillna(0)
        if 'pilot/throttle' in df.columns:
            df.loc[df['user/throttle'] == 0, 'user/throttle'] = df['pilot/throttle'].fillna(0)

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

def get_dataframe_from_db( input_file, conn, sources: list=None, image_min: int=128 ):
    """ Load DonkeyCar training data from a database and return a dataframe.
           The database is created in malpi/dk/scripts/tub2db.py.
           sources: a list of Tub names to be used directly in the select statement
               otherwise get Tub names from all files listed in input_file (list or string)
               if sources and input_file are both none then return all sources.
           image_min: Minimum width and height for images.
    """

    if sources is None and input_file is None:
        names = ""
    elif input_file is not None:
        if isinstance(input_file, str):
            input_file = [input_file]
        filelist = preprocessFileList( input_file )
        names = [ f'"{Path(f).name}"' for f in filelist if '"' not in f ]
        names = f'AND Sources.name in ({", ".join(names)})'
    else:
        names = [ f'"{s}"' for s in sources ]
        names = f'AND Sources.name in ({", ".join(names)})'

    if image_min is not None:
        images = f"""AND Sources.image_width >= {image_min}
    AND Sources.image_height >= {image_min}"""
    else:
        images = ""

# Edited values for angle and throttle override all others.
# Otherwise user overrides pilot. But there's no way to know if the user overrode the pilot if the user value is zero.
    sql=f"""SELECT Sources.full_path || '/' || '{Tub.images()}' || '/' || TubRecords.image_path as "cam/image_array",
-- edit > user > pilot
case when edit_angle is not null then edit_angle
     when pilot_angle is not null and user_angle == 0.0 then pilot_angle
     else user_angle end as "user/angle",
case when edit_throttle is not null then edit_throttle
     when pilot_throttle is not null and user_throttle == 0.0 then pilot_throttle
     else user_throttle end as "user/throttle"
  FROM TubRecords, Sources
 WHERE TubRecords.source_id = Sources.source_id
{names}
{images}
AND TubRecords.deleted = 0;"""

    df = pd.read_sql_query(sql, conn)

    return df

def get_dataframe_from_db_with_aux( input_file, conn, sources: list=None, image_min: int=128 ):
    """ Load DonkeyCar training data from a database and return a dataframe.
           The database is created in malpi/dk/scripts/tub2db.py.
           sources: a list of Tub names to be used directly in the select statement
               otherwise get Tub names from all files listed in input_file (list or string)
               if sources and input_file are both none then return all sources.
           image_min: Minimum width and height for images.
    """

    if sources is None and input_file is None:
        names = ""
    elif input_file is not None:
        if isinstance(input_file, str):
            input_file = [input_file]
        filelist = preprocessFileList( input_file )
        names = [ f'"{Path(f).name}"' for f in filelist if '"' not in f ]
        names = f'AND Sources.name in ({", ".join(names)})'
    else:
        names = [ f'"{s}"' for s in sources ]
        names = f'AND Sources.name in ({", ".join(names)})'

    if image_min is not None:
        images = f"""AND Sources.image_width >= {image_min}
    AND Sources.image_height >= {image_min}"""
    else:
        images = ""

# Edited values for angle and throttle override all others.
# Otherwise user overrides pilot. But there's no way to know if the user overrode the pilot if the user value is zero.
# select t.source_id, pos_cte, sm.value, tr.track_id from TubRecords t, Sources s, SourceMeta sm, Tracks tr where t.source_id = s.source_id and s.source_id = sm.source_id and sm.key="DONKEY_GYM_ENV_NAME" AND sm.value=tr.gym_name ORDER BY RANDOM() LIMIT 10;

    sql=f"""SELECT Sources.full_path || '/' || '{Tub.images()}' || '/' || TubRecords.image_path as "cam/image_array",
        case when edit_angle is not null then edit_angle
             when pilot_angle is not null and user_angle == 0.0 then pilot_angle
             else user_angle end as "user/angle",
        case when edit_throttle is not null then edit_throttle
             when pilot_throttle is not null and user_throttle == 0.0 then pilot_throttle
             else user_throttle end as "user/throttle",
           TubRecords.pos_cte,
           Tracks.track_id
  FROM TubRecords, Sources, SourceMeta, Tracks
 WHERE TubRecords.source_id = Sources.source_id
   AND Sources.source_id = SourceMeta.source_id
   AND SourceMeta.key = "DONKEY_GYM_ENV_NAME"
   AND SourceMeta.value = Tracks.gym_name
   AND TubRecords.pos_cte is not null
 {names}
 {images}
AND TubRecords.deleted = 0;"""

    df = pd.read_sql_query(sql, conn)
    df['user/angle'] = df['user/angle'].astype(np.float32)
    df['user/throttle'] = df['user/throttle'].astype(np.float32)
    df['pos_cte'] = df['pos_cte'].astype(np.float32)
    df['track_id'] = df['track_id'].astype(np.int64)

    return df

def get_track_metadata( conn ):
    # Get all track meta data and return it as a dictionary
    sql = """SELECT track_id, gym_name, sim_name FROM Tracks;"""
    c = conn.cursor()
    c.execute(sql)
    track_meta = {t: (a,b) for t,a,b in c.fetchall()}

    return track_meta

def get_data(inputs, df_all=None, batch_tfms=None, item_tfms=None, verbose=False, autoencoder=False, aux=False):

    if df_all is None:
        df_all = get_dataframe(inputs, verbose)

    if item_tfms is None:
        tfms = [Resize(128,method="squish")]
    else:
        tfms = item_tfms

    if autoencoder:
        if aux:
            blocks = (ImageBlock, ImageBlock, RegressionBlock(n_out=3), CategoryBlock)
            y_reader = [ColReader("cam/image_array"), 
                        ColReader( ["user/angle", "user/throttle", "pos_cte"] ),
                        ColReader("track_id")]
        else:
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

def get_prevae_data( vae_model_file, dls=None, input_file="tracks_all.txt", z_len=64, verbose=False ):
    """ Replace DonkeyCar image data with mu/log_var from a pre-trained VAE.
            Load data from the tubs listed in input_file
            Use the VAE to generate mu/log_var
            Return those along with throttle and steering
    """

    if dls is None:
        df_all = get_dataframe(input_file)
        dls = get_data(input_file, df_all=df_all, verbose=False, autoencoder=False)

    learn = load_learner(vae_model_file, cpu=False)

    mus = None
    var = None
    outputs = None
    total = 0
    learn.model.eval()
    with torch.no_grad():
        for images, controls in dls.train:
            total += images.shape[0]
            _, _, mu, log_var = learn.forward( images)
            if mus is None:
                mus = mu
            else:
                mus = torch.cat( (mus, mu), 0)
            if var is None:
                var = log_var
            else:
                var = torch.cat( (var, log_var), 0)
            if outputs is None:
                outputs = controls
            else:
                outputs = torch.cat( (outputs, controls), 0)
        for images, controls in dls.valid:
            total += images.shape[0]
            _, _, mu, log_var = learn.forward( images)
            if mus is None:
                mus = mu
            else:
                mus = torch.cat( (mus, mu), 0)
            if var is None:
                var = log_var
            else:
                var = torch.cat( (var, log_var), 0)
            if outputs is None:
                outputs = controls
            else:
                outputs = torch.cat( (outputs, controls), 0)

    if verbose:
        print( f"Mus: {type(mus)} {mus.shape}" )
        print( f"var: {type(var)} {var.shape}" )
        print( f"outputs: {type(outputs)} {outputs.shape}" )

#Create a new dataframe from mu/log_var from the vae and steering/throttle
    df_new = pd.DataFrame()
    df_new['mu'] = np.array(mus.cpu()).tolist()
    df_new['var_log'] = np.array(var.cpu()).tolist()
    df_new['user/angle'] = np.array(outputs[:,0].cpu())
    df_new['user/throttle'] = np.array(outputs[:,1].cpu())

# Save to sqlite? https://kontext.tech/article/633/pandas-save-dataframe-to-sqlite

#    df_256 = df_all[['user/angle','user/throttle']][0:mus.shape[0]].copy()
#    df_256['mu'] = np.array(mus.cpu()).tolist()
#    df_256['var_log'] = np.array(var.cpu()).tolist()

    blocks = (RegressionBlock(n_out=z_len), RegressionBlock(n_out=z_len), RegressionBlock(n_out=2))
    y_reader = ColReader(['user/angle','user/throttle'])
    pascal = DataBlock(blocks=blocks,
                       splitter=RandomSplitter(),
                       get_x=[ColReader("mu"),ColReader("var_log")],
                       get_y=y_reader,
                       item_tfms=None,
                       batch_tfms=None,
                       n_inp=2)

    dls = pascal.dataloaders(df_new)

    return dls
