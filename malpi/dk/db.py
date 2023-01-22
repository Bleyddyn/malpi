""" Create an sqlite3 data for holding DonkeyCar Tub data.

    Currently only supports sqlite3.
    Currently only supports v2 Tubs.
    Images are not stored in the database, only the path relative to the Tub's location.
"""

import argparse
import sqlite3

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
    cur.execute("DROP TABLE IF EXISTS TubStaging;")

# Move insert_one_tub from malpi/dk/scripts/tub2db.py to here?
#def insert_one_tub(conn, tub):

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create an empty sqlite3 database for DonkeyCar Tub data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('database', help="An sqlite3 database (will be created if it doesn't exist")

    args = parser.parse_args()

    database = args.database
    conn = sqlite3.connect(database)
    create_db(conn)

    conn.close()
