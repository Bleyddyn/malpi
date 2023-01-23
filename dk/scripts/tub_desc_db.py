#! /usr/bin/env python3
"""
Describe Sources (Tubs) found in a database
"""

import os
# suppress warning from numexpr module
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import sys
import time
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

# Import DonkeyCar, suppressing it's annoying banner
from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    import donkeycar as dk

"""
--CREATE TABLE IF NOT EXISTS "Sources" (
--    "source_id" INTEGER PRIMARY KEY NOT NULL,
--    "name" TEXT NOT NULL,
--    "full_path" TEXT NOT NULL,
--    "version" REAL DEFAULT 2.0, -- DonkeyCar Tub version
--    "image_width" INTEGER,
--    "image_height" INTEGER,
--    "created" REAL,
--    "count" INTEGER,
--    "inputs" TEXT,
--    "types" TEXT,
--    UNIQUE(name, full_path)
--    );
--sqlite> .schema TubRecords
--CREATE TABLE IF NOT EXISTS "TubRecords" (
--    "source_id" INTEGER NOT NULL,
--    "tub_index" INTEGER NOT NULL,
--    "timestamp_ms" INTEGER,
--    "image_path" TEXT,
--    "mode" TEXT,
--    "user_angle" REAL,
--    "user_throttle" REAL,
--    "pilot_angle" REAL,
--    "pilot_throttle" REAL,
--    "edit_angle" REAL, --Manually edited
--    "edit_throttle" REAL, --Manually edited
--    "deleted" BOOLEAN NOT NULL DEFAULT false CHECK (deleted IN (0, 1)),
--    PRIMARY KEY(source_id, tub_index),
--    FOREIGN KEY(source_id) REFERENCES Sources (source_id)
--       ON UPDATE CASCADE
--       ON DELETE RESTRICT
--    );
--sqlite> 
--
-- Language: sql
-- Select all TubRecords for a given key/value pair,

SELECT image_path, 
-- Add a case statement to get pilot_angle and pilot_throttle if not null
-- otherwise use user_angle and user_throttle
case when pilot_angle is not null then pilot_angle else user_angle end as angle,
case when pilot_throttle is not null then pilot_throttle else user_throttle end as throttle
  FROM TubRecords, Sources, SourceMeta
 WHERE SourceMeta.key="DONKEY_GYM_ENV_NAME" AND SourceMeta.value="donkey-warehouse-v0"
AND TubRecords.source_id = Sources.source_id
AND Sources.source_id = SourceMeta.source_id
AND TubRecords.deleted = 0
ORDER BY RANDOM() LIMIT 10;
"""

class EmptyTub:
    def __init__(self, base_path):
        self.meta = dict()
        self.path = base_path
        self.base_path = base_path

        manifest_path = Path(os.path.join(self.base_path, 'manifest.json'))
        if not manifest_path.exists():
            raise ValueError(f"No manifest.json in {base_path}")

        with open(manifest_path, 'r') as f:
            self.inputs = json.loads(f.readline()) # inputs
            self.types = json.loads(f.readline()) # types
            self.meta = json.loads(f.readline())

    def __len__(self):
        return 0

def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessFileList( filelist ):
    dirs = []
    if filelist is not None:
        for afile in filelist:
            afile = os.path.expanduser(afile)
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    removeComments( dirs )
    return dirs

def describe_source( source, conn, time_of_day=False, meta=[], img=False ):
    """ TODO: This should be generalized to return only user-requested meta data.
        TODO: Add a check for image sizes
 Tub     Version # Samples       Location        Task    Driver  Throttle        Steering        DONKEY_GYM_ENV_NAME
20200518_16.tub v1      900     sim     Race    Andrew  1.0     1.0     donkey-mountain-track-v0
20200518_17.tub v1      1401    sim     Race    Andrew  1.0     1.0     donkey-mountain-track-v0
--CREATE TABLE IF NOT EXISTS "Sources" (
--    "source_id" INTEGER PRIMARY KEY NOT NULL,
--    "name" TEXT NOT NULL,
--    "full_path" TEXT NOT NULL,
--    "version" REAL DEFAULT 2.0, -- DonkeyCar Tub version
--    "image_width" INTEGER,
--    "image_height" INTEGER,
--    "created" REAL,
--    "count" INTEGER,
--    "inputs" TEXT,
--    "types" TEXT,
--    UNIQUE(name, full_path)

CREATE TABLE IF NOT EXISTS "SourceMeta" (
    "source_id" INTEGER NOT NULL,
    "key" TEXT,
    "value" TEXT,
    PRIMARY KEY(source_id, key),
    FOREIGN KEY(source_id) REFERENCES Sources (source_id)
       ON UPDATE CASCADE
       ON DELETE RESTRICT
    );

    """

    version = source['version']
    count = source['count']
    base_path = source['name']
    tod = source['created']

    tub_inputs = {}

# Get all rows from the SourceMeta table for this source and create a dictionary
    c = conn.cursor()
    c.execute("SELECT key, value FROM SourceMeta WHERE source_id = ?", (source['source_id'],))
    tub_meta = dict(c.fetchall())

    loc = tub_meta.get("location", "NA")
    task = tub_meta.get("task", "NA")
    driver = tub_meta.get("driver", "NA")
    throttle = tub_meta.get("JOYSTICK_MAX_THROTTLE", "NA")
    steering = tub_meta.get("JOYSTICK_STEERING_SCALE", "NA")

# Parse source inputs json into a dictionary
    if source['inputs'] is not None:
        tub_inputs = json.loads(source['inputs'])

    if time_of_day:
        if tod is not None:
            tod = int(tod)
            tod = datetime.fromtimestamp(tod).strftime('\t%H:%M')
        else:
            tod = "\t"
    else:
        tod = ""

    meta_st = ""
    for key in meta:
        if key in tub_inputs:
            meta_st += "\tInput"
        elif key in tub_meta:
            meta_st += "\t{}".format( tub_meta[key] )
        else:
            meta_st += "\tNo"

    img_st = ""
    if img:
        img_st = "\t{}".format( (source['image_width'], source['image_height']) )

    print( "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}{}{}{}".format( source['name'].strip(), version, count, loc, task, driver, throttle, steering, tod, meta_st, img_st ) )
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Describe Tub records in a database. Pipe through "| tail -n +10 | column -t -s $\'\\t\'" to display in columns.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tod', action='store_true', default=False, help='Display time of day for beginning of each tub file.')
    parser.add_argument('--img', action='store_true', default=False, help='Display image size.')
    #parser.add_argument('-t', '--tub', nargs="*", help='Name of a tub to be described. Option may be used more than once.')
    parser.add_argument('--meta', nargs='*', default=["DONKEY_GYM_ENV_NAME"], help='One or more meta keys to search for in Tub files.')
    parser.add_argument('db', nargs=1, default=["tubs.sqlite"], help='Database file containing Tub data.')

    args = parser.parse_args()

# Open the database
    db = args.db[0]
    if not os.path.exists(db):
        print( f"Database file {db} not found." )
        exit(1)


    done = []
    counts = []
    stat_str = ""
    if args.tod:
        tod_header = "\tTime"
    else:
        tod_header = ""
    meta_header = ""
    for meta in args.meta:
        meta_header += "\t" + meta

    img_header = ""
    if args.img:
        img_header = "\tImg Size"

    print( "\nTub\tVersion\t# Samples\tLocation\tTask\tDriver\tThrottle\tSteering{}{}{}{}".format( stat_str, tod_header, meta_header, img_header ) )

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    # Fetch every entry in the Sources table
    c = conn.cursor()
    c.execute("SELECT * FROM Sources ORDER BY created ASC")
    rows = c.fetchall()
    for row in rows:
        cnt = describe_source(row, conn, time_of_day=args.tod, meta=args.meta, img=args.img)
        counts.append( cnt )

    print()
    print( "Total samples: {}".format( np.sum(counts) ) )
