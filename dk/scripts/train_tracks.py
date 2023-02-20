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

import argparse
import sqlite3
from datetime import datetime
import json

from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import rmse
import torch
from torch import nn

from donkeycar.parts.tub_v2 import Tub
import pandas as pd
from pathlib import Path

from malpi.dk.train import preprocessFileList, get_data, get_learner, get_autoencoder, train_autoencoder
from malpi.dk.vae import VanillaVAE, SplitDriver
from malpi.dk.train import get_dataframe, get_prevae_data, get_dataframe_from_db

from gym_test import main as gym_test

def get_pilot_learner( dls, z_len, verbose=False ):
    pmodel = SplitDriver(z_len)
    if verbose:
        print(pmodel)
    callbacks=ActivationStats(with_hist=True)
    learn = Learner(dls, pmodel,  loss_func=torch.nn.MSELoss(), metrics=[rmse], cbs=callbacks)
    return learn

def train_dk( input_file, epochs, lr, name, verbose=True ):
    dls = get_data(input_file, verbose=True)
    learn = get_learner(dls)
    learn.fit_one_cycle(epochs, lr)
    learn.export( name + ".pkl" )
    learn.recorder.plot_loss()
    #learn.show_results(figsize=(20,10))
    plt.savefig(name + '.png')

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

def print_results(results):
    # Outer loop is a list of drivers, inner loop is a dictionary of tracks
    # See malpi/dk/scripts/gym_test.py
    driver_name = None
    for driver in results:
        one_driver = driver['driver']
        if driver_name is None:
            print( f"Driver {one_driver}" )
            driver_name = one_driver
        elif driver != one_driver:
            print( f"Driver {one_driver}" )
        for track_name in driver.keys():
            if track_name == 'driver':
                continue
            track = driver[track_name]
            print( f"     track: {track_name}" )
            print( f"      laps: {track['lap_times']}" )
            print( f"     steps: {track['steps']}" )
            print( f"   rewards: {track['rewards']}" )
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on each input file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--vae_model', type=str, default=None,help='File with a pre-trained VAE model. If not specified, train a non VAE model.')
    parser.add_argument('--database', type=str, default=None,help='Path to an sqlite3 database.')
    parser.add_argument('--one_model', action='store_true', default=False, help='Train a single model on all the input files.')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('input_files', nargs='+', help='List of input files (each of which has a list of tubs)')
    args = parser.parse_args()

    vae_model = args.vae_model # "models/vae_v3.pkl"
    epochs = args.epochs
    lr = args.lr
    callbacks = [EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=5)]

    results = []

    if args.one_model:
        dt_string = datetime.now().strftime("%Y%m%d_%H%M")
        output_name = args.output_dir + '/all_' + dt_string + '.pkl'

        if args.database is not None:
            conn = sqlite3.connect(args.database)
            df = get_dataframe_from_db( None, conn ) # Get all the data
            print( f"Training on all tracks with {len(df)} records from the database" )
            dls = get_data( None, df_all=df, verbose=True)
            conn.close()
        else:
            dls = get_data(args.input_files, verbose=False)

        # Get the learner
        if vae_model is not None:
            dls = get_prevae_data( vae_model, dls=dls, input_file=None, verbose=False )
            learn = get_pilot_learner( dls, z_len=128, verbose=False )
        else:
            print( f"Training data: {len(dls.train_ds)} / {len(dls.valid_ds)}" )
            learn = get_learner(dls)

        # Train and save the model
        learn.fit_one_cycle(epochs, lr, cbs=callbacks)
        learn.export( output_name )

        # Test the model on all tracks
        res = gym_test('all', output_name, vae_model=vae_model)
        results.append( res )

    else:
        for input_file in args.input_files:
            track_name = get_track_name(input_file)
            #print( f"Training on track {track_name} from file {input_file}" )
            output_name = args.output_dir + '/' + track_name + '.pkl'

            # Get the data
            if args.database is not None:
                conn = sqlite3.connect(args.database)
                df = get_dataframe_from_db( input_file, conn )
                print( f"Training on track {track_name} from file {input_file} with {len(df)} records from the database" )
                dls = get_data( input_file, df_all=df, verbose=True)
                conn.close()
            else:
                dls = get_data(input_file, verbose=False)

            # Get the learner
            if vae_model is not None:
                dls = get_prevae_data( vae_model, dls=dls, input_file=input_file, verbose=False )
                learn = get_pilot_learner( dls, z_len=128, verbose=False )

            else:
                print( f"Training data: {len(dls.train_ds)} / {len(dls.valid_ds)}" )
                learn = get_learner(dls)

            # Train and save the model
            learn.fit_one_cycle(epochs, lr, cbs=callbacks)
            learn.export( output_name )

            # Test the model
            res = gym_test(track_name, output_name, vae_model=vae_model)
            results.append( res )

    # Export the results to json
    with open(args.output_dir + '/results.json', 'w') as f:
        json.dump(results, f)
    print_results(results)
