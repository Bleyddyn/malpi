""" Test MaLPi's database module. """

import pytest
import sqlite3

from malpi.dk.db import create_db
from malpi.dk.train import get_dataframe_from_db

"""sqlite> .schema TubRecords
CREATE TABLE IF NOT EXISTS "TubRecords" (
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
    );
sqlite> .schema Sources
CREATE TABLE IF NOT EXISTS "Sources" (
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
    );
"""

def create_test_db():
    """ Create a database for testing. """
    created = 1642635551.96579 # Arbitrary timestamp
    count = 5 # Number of records, should match the number of records in the test data

    conn = sqlite3.connect('file:cachedb?mode=memory&cache=shared')
    create_db(conn)

    cur = conn.cursor()
    cur.execute('SELECT * FROM TubRecords;')
    rows = cur.fetchall()
    if len(rows) == 5:
        return conn

    conn.execute('INSERT INTO Sources (name, full_path, image_width, image_height, created, count, inputs, types) VALUES (?, ?, ?, ?, ?, ?, ?, ?);', ('TestFilePleaseIgnore', '/tmp/dummy.tub', 128, 128, created, count, '[input1, input2]', '[float,int]'))

    insert_rec = 'INSERT INTO TubRecords (source_id, tub_index, timestamp_ms, image_path, mode, user_angle, user_throttle, pilot_angle, pilot_throttle, edit_angle, edit_throttle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'

    # Insert count records into the TubRecords table
    conn.execute(insert_rec, (1, 1, 1642635552256, 'dummy1.jpg', 'user', 0.0, 0.0, 0.0, 0.0, None, None))
    conn.execute(insert_rec, (1, 2, 1642635552356, 'dummy2.jpg', 'user', 0.1, 0.2, 0.0, 0.0, None, None))
    conn.execute(insert_rec, (1, 3, 1642635552456, 'dummy3.jpg', 'pilot/angle', 0.0, 0.0, 1.1, 1.2, None, None))
    conn.execute(insert_rec, (1, 4, 1642635552556, 'dummy4.jpg', 'pilot/angle', 0.1, 0.2, 1.1, 1.2, None, None))
    conn.execute(insert_rec, (1, 5, 1642635552656, 'dummy5.jpg', 'user', 0.1, 0.2, 1.1, 1.2, 0.4, 0.5))

    conn.commit()

    return conn

# A pytest setup method to create a database for testing and fill it with test data.
@pytest.fixture
def db():
    return create_test_db()

# Test to make sure the TubRecords table was created correctly
def test_tubrecords(db):
    """ Test the TubRecords table. """
    cur = db.cursor()
    cur.execute('SELECT * FROM TubRecords')
    rows = cur.fetchall()
    cur.close()
    assert len(rows) == 5

def test_user_mode1(db):
    df = get_dataframe_from_db( None, db, sources=["TestFilePleaseIgnore"] )
    assert len(df) == 5
    assert list(df[ ['user/angle','user/throttle'] ].loc[0].values) == [0.0,0.0]

def test_user_mode2(db):
    df = get_dataframe_from_db( None, db, sources=["TestFilePleaseIgnore"] )
    assert list(df[ ['user/angle','user/throttle'] ].loc[1].values) == [0.1,0.2]

def test_user_mode3(db):
    df = get_dataframe_from_db( None, db, sources=["TestFilePleaseIgnore"] )
    assert list(df[ ['user/angle','user/throttle'] ].loc[2].values) == [1.1,1.2]

def test_user_mode4(db):
    df = get_dataframe_from_db( None, db, sources=["TestFilePleaseIgnore"] )
    assert list(df[ ['user/angle','user/throttle'] ].loc[3].values) == [0.1,0.2]

def test_user_mode5(db):
    df = get_dataframe_from_db( None, db, sources=["TestFilePleaseIgnore"] )
    assert list(df[ ['user/angle','user/throttle'] ].loc[4].values) == [0.4,0.5]

