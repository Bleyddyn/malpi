""" Transfer a DonkeyCar Tub file to a database.

    Currently only supports sqlite3.
    Currently only supports v2 Tubs.
    Images are not stored in the database, only the path relative to the Tub's location.

    Example of how to use json in sqlite3 (see: https://www.sqlite.org/json1.html):
    sqlite> select inputs from Sources;
        ["cam/image_array", "user/angle", "user/throttle", "user/mode", "timestamp"]
    sqlite> select json_extract(inputs,'$[0]') from Sources;
        cam/image_array

"""

import argparse
import pathlib
import json
import sqlite3
import pandas as pd
from PIL import Image

from donkeycar.parts.tub_v2 import Tub

def create_track_table(con):
    """
track_id gym_name sim_name
1 donkey-warehouse-v0 warehouse
2 donkey-generated-roads-v0 generated_road
3 donkey-avc-sparkfun-v0 sparkfun_avc
4 donkey-generated-track-v0 generated_track
5 donkey-roboracingleague-track-v0	roboracingleague_1
6 donkey-waveshare-v0 waveshare
7 donkey-minimonaco-track-v0 mini_monaco
8 donkey-mountain-track-v0 mountain_track
9 donkey-warren-track-v0 warren
10 donkey-circuit-launch-track-v0 circuit_launch
"""
    track_string='''CREATE TABLE IF NOT EXISTS "Tracks" (
    "track_id" INTEGER PRIMARY KEY NOT NULL,
    "gym_name" TEXT NOT NULL,
    "sim_name" TEXT,
    UNIQUE(gym_name)
    );'''

    cur = con.cursor()
    cur.execute(track_string)

    cur.execute("SELECT COUNT(*) AS CNTREC FROM Tracks;")
    row=cur.fetchone()
    if row[0] == 0:
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (1, 'donkey-warehouse-v0', 'warehouse');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (2, 'donkey-generated-roads-v0', 'generated_road');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (3, 'donkey-avc-sparkfun-v0', 'sparkfun_avc');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (4, 'donkey-generated-track-v0', 'generated_track');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (5, 'donkey-roboracingleague-track-v0', 'roboracingleague_1');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (6, 'donkey-waveshare-v0', 'waveshare');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (7, 'donkey-minimonaco-track-v0', 'mini_monaco');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (8, 'donkey-mountain-track-v0', 'mountain_track');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (9, 'donkey-warren-track-v0', 'warren');")
        cur.execute("INSERT INTO Tracks (track_id, gym_name, sim_name) VALUES (10, 'donkey-circuit-launch-track-v0', 'circuit_launch');")

def create_db(con):
    tub_records_string='''CREATE TABLE IF NOT EXISTS "TubRecords" (
    "source_id" INTEGER NOT NULL,
    "tub_index" INTEGER NOT NULL,
    "timestamp_ms" INTEGER,
    "image_path" TEXT,
    "mode" TEXT,
    "user_angle" REAL,
    "user_throttle" REAL,
    "pilot_angle" REAL,
    "pilot_throttle" REAL,
    "edit_angle" REAL, --Manually edited
    "edit_throttle" REAL, --Manually edited
    "deleted" BOOLEAN NOT NULL DEFAULT false CHECK (deleted IN (0, 1)),
    "pos_cte" REAL DEFAULT NULL,
    "pos_pos_x" REAL DEFAULT NULL,
    "pos_pos_y" REAL DEFAULT NULL,
    "pos_pos_z" REAL DEFAULT NULL,
    "pos_speed" REAL DEFAULT NULL,
    PRIMARY KEY(source_id, tub_index),
    FOREIGN KEY(source_id) REFERENCES Sources (source_id)
       ON UPDATE CASCADE
       ON DELETE RESTRICT
    );'''

    sources_string='''CREATE TABLE IF NOT EXISTS "Sources" (
    "source_id" INTEGER PRIMARY KEY NOT NULL,
    "name" TEXT NOT NULL,
    "full_path" TEXT NOT NULL,
    "version" REAL DEFAULT 2.0, -- DonkeyCar Tub version
    "image_width" INTEGER,
    "image_height" INTEGER,
    "created" REAL,
    "count" INTEGER,
    "inputs" TEXT,
    "types" TEXT,
    UNIQUE(name, full_path)
    );'''

    source_meta='''CREATE TABLE IF NOT EXISTS "SourceMeta" (
    "source_id" INTEGER NOT NULL,
    "key" TEXT,
    "value" TEXT,
    PRIMARY KEY(source_id, key),
    FOREIGN KEY(source_id) REFERENCES Sources (source_id)
       ON UPDATE CASCADE
       ON DELETE RESTRICT
    );'''

    cur = con.cursor()
    cur.execute(sources_string)
    cur.execute(source_meta)
    cur.execute(tub_records_string)
    create_track_table(con)
    cur.execute("DROP TABLE IF EXISTS TubStaging;")

    # See https://stackoverflow.com/questions/18920136/check-if-a-column-exists-in-sqlite
    pos_query="SELECT COUNT(*) AS CNTREC FROM pragma_table_info('TubRecords') WHERE name='pos_cte';"
    has_cols=cur.execute(pos_query)
    row=has_cols.fetchone()
    if row[0] == 0:
        cur.execute("ALTER TABLE TubRecords ADD COLUMN pos_cte REAL DEFAULT NULL")
        cur.execute("ALTER TABLE TubRecords ADD COLUMN pos_pos_x REAL DEFAULT NULL")
        cur.execute("ALTER TABLE TubRecords ADD COLUMN pos_pos_y REAL DEFAULT NULL")
        cur.execute("ALTER TABLE TubRecords ADD COLUMN pos_pos_z REAL DEFAULT NULL")
        cur.execute("ALTER TABLE TubRecords ADD COLUMN pos_speed REAL DEFAULT NULL")

def get_training(conn):
    # See: https://www.sqlitetutorial.net/sqlite-case/
    # select "cam/image_array", CASE WHEN "user/angle" != 0 THEN "user/angle" WHEN "pilot/angle" != 0 THEN "pilot/angle" ELSE 0 END angle from TubRecords limit 10;
    pass

def tub_to_staging(conn, tub):
    df = pd.DataFrame(tub)
    # Insert the dataframe into a staging table
    df.to_sql(name="TubStaging", con=conn, schema=None, if_exists='replace', index=False, index_label="df_index", chunksize=None, dtype=None, method=None)
    return df

def update1(conn, source_name):
    """ NOTE: I'm not going to try this for now. Updating will require deleting the db and reloading everything.
        Update new columns from staging table to TubRecords table.
        pos/cte, pos/pos_x, pos/pos_y, pos/pos_z, pos/speed """

    """ From: https://stackoverflow.com/questions/28668817/update-column-with-value-from-another-table-using-sqlite
        UPDATE table1 
           SET status = (SELECT t2.status FROM table2 t2 WHERE t2.trans_id = id)
           WHERE id IN (SELECT trans_id FROM table2 t2 WHERE t2.trans_id= id)
           """

    cur = conn.cursor()
    cur.execute( "UPDATE TubRecords SET pos_cte = (SELECT 't2.pos/cte' FROM TubStaging t2 WHERE t2.df_index = tub_index) WHERE tub_index IN (SELECT t2.df_index FROM TubStaging t2 WHERE t2.df_index= tub_index);" )
    cur.execute("UPDATE TubRecords SET pos_pos_x = (SELECT pos_pos_x FROM TubStaging WHERE TubStaging.df_index = TubRecords.df_index);")
    cur.execute("UPDATE TubRecords SET pos_pos_y = (SELECT pos_pos_y FROM TubStaging WHERE TubStaging.df_index = TubRecords.df_index);")
    cur.execute("UPDATE TubRecords SET pos_pos_z = (SELECT pos_pos_z FROM TubStaging WHERE TubStaging.df_index = TubRecords.df_index);")
    cur.execute("UPDATE TubRecords SET pos_speed = (SELECT pos_speed FROM TubStaging WHERE TubStaging.df_index = TubRecords.df_index);")

def insert_one_tub(conn, tub):

    df = tub_to_staging(conn, tub)

    # get image width/height
    img = Image.open( tub.images_base_path + "/0_cam_image_array_.jpg" )
    image_width = img.width
    image_height = img.height
    created = tub.manifest.manifest_metadata['created_at']
    count = len(tub) # tub.manifest.current_index

    path_path = pathlib.Path(tub.base_path)
    full_path = str(path_path.resolve())
    fname = str(path_path.name)

    # It might be better to store these as a single dictionary of (input,type) items.
    # Wait and see if/how I ever use this
    inputs = json.dumps(tub.manifest.inputs)
    types = json.dumps(tub.manifest.types)


    new_source="""insert into Sources (name, full_path, image_width, image_height, created, count, inputs, types)
                             values (?,?,?,?,?,?,?,?);"""
    source_meta="insert into SourceMeta (source_id, key, value) values (?,?,?);"
# Need to handle pilot/angle and edit/angle (if present)
# Check if the dataframe has a column named "pilot/angle"

    extra_columns = [ f'"{c}"' for c in df.columns if c in ["pilot/angle", "pilot/throttle",
                                      "pos/cte", "pos/pos_x", "pos/pos_y", "pos/pos_z", "pos/speed"]]
    if len(extra_columns) > 0:
        extra_columns_string = ", " + ", ".join(extra_columns)
    else:
        extra_columns_string = ""
    # create a copy of extra_columns with '/' replaced by '_'
    extra_underscore = extra_columns_string.replace('/','_')

    stage=f'''insert into TubRecords (source_id, tub_index, timestamp_ms, image_path, mode, user_angle,
                                      user_throttle{extra_underscore})
                              select ?, _index, _timestamp_ms, "cam/image_array", "user/mode", "user/angle",
                                      "user/throttle"{extra_columns_string} from TubStaging;'''

    cur = conn.cursor()

    # Create a new entry in the Sources table for this Tub file, and add meta data to the SourceMeta table
    try:
        res = cur.execute( new_source, (fname, full_path, image_width, image_height, created, count, inputs, types) );
        source = res.lastrowid
    except sqlite3.IntegrityError as ex:
        # E.g. sqlite3.IntegrityError: UNIQUE constraint failed: Sources.name, Sources.full_path
        # For the moment assume this means this tub has already been loaded
        print( f"Skipping. Tub has already been loaded: {fname}" )
        print( f"   Exception was: {ex}" )
        return

    for meta in tub.manifest.metadata.items():
        cur.execute( source_meta, (source, meta[0], str(meta[1])) );

    # Copy desired fields from staging to TubRecords
    try:
        cur.execute( stage, (source,) )
    except sqlite3.IntegrityError as ex:
        print( f"Skipping. Duplicate records in Tub: {fname}" )
        return

    conn.commit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Import a DonkeyCar Tub file into an sqlite3 database', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('database', help="An sqlite3 database (will be created if it doesn't exist")
    parser.add_argument('tub_path', nargs='*', help='Path to a DonkeyCar Tub file')
    parser.add_argument('--gen_tracks', action='store_true', default=False, help='Create the Tracks table, then exit')

    args = parser.parse_args()

    database = args.database
    conn = sqlite3.connect(database)
    create_db(conn)

    if args.gen_tracks:
        create_track_table(conn)
        conn.commit()
        exit(0)

    for fname in args.tub_path:
        try:
            tub = Tub(fname, read_only=True)
        except FileNotFoundError as ex:
            print( f"Skipping. Unable to open tub: {fname}" )
            continue
        insert_one_tub(conn, tub)

    # Commit all of the insert statements
    conn.close()
