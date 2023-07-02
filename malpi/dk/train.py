import os
from pathlib import Path
import sqlite3

import torch

from donkeycar.parts.tub_v2 import Tub
import pandas as pd
import numpy as np


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

""" Not being used. Includes Fastai layers.
def get_base_dk_model(dls):
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
    return model
"""
