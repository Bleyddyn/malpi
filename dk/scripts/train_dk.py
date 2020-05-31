#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Uses the data written by the donkey v2.2 tub writer,
but faster training with proper sampling of distribution over tubs. 
Has settings for continuous training that will look for new files as it trains. 
Modify on_best_model if you wish continuous training to update your pi as it builds.
You can drop this in your ~/mycar dir.
Basic usage should feel familiar: python train.py --model models/mypilot
The --email option defaults to None (docopt doesn't seem to handle that case explicitly)

Usage:
    train.py [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|latent|categorical|rnn|imu|behavior|3d|look_ahead|tensorrt_linear|tflite_linear|coral_tflite_linear)] [--continuous] [--aug] [--email=<email>] [--name=<exp_name>] [--exp=<exp_dir>]

Options:
    -h --help         Show this screen.
    -f --file=<file>  A text file containing paths to tub files, one per line. Option may be used more than once.
    --email=<email>   Email address to send training finished notifications. Requires email configuration file.
    --name=<exp_name> Name for this training experiment.
    --exp=<exp_dir>   Top level experiment directory. [default: experiments]
"""

import os
from docopt import docopt

import numpy as np
import tensorflow as tf

import donkeycar as dk
from donkeycar.train.train import preprocessFileList, train, make_model, Generators, plot_history
from donkeycar.utils import ImageDim
from malpi.notify import notify, read_email_config
from malpi import Experiment

if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()
    tub = args['--tub']
    model = args['--model']
    transfer = args['--transfer']
    model_type = args['--type']
    continuous = args['--continuous']
    aug = args['--aug']
    
    dirs = preprocessFileList( args['--file'] )
    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend( tub_paths )

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE

    image_dim = None
    if hasattr(cfg, 'IMAGE_MODEL_W') and hasattr(cfg, 'IMAGE_MODEL_H'):
        cfg.IMAGE_W = cfg.IMAGE_MODEL_W
        cfg.IMAGE_H = cfg.IMAGE_MODEL_H
        cfg.TARGET_W = cfg.IMAGE_MODEL_W
        cfg.TARGET_H = cfg.IMAGE_MODEL_H
        cfg.TARGET_D = cfg.IMAGE_DEPTH

    kl, model_name = make_model(cfg, model, transfer, model_type, image_dim=image_dim )

    gens = Generators( cfg, dirs, kl, model_type, continuous, aug=aug )

    cfg.model_type = model_type

    print("Samples: total: %d, train: %d, val: %d" % (gens.total_count(), gens.train_count(), gens.val_count()))
    print('Steps per epoch: %d' % gens.train_steps())

    exp = None
    if args['--name'] is not None:
        args['dirs'] = dirs
        exp = Experiment( args['--name'], args, exp_dir=args['--exp'], num_samples=gens.total_count(), input_dim=(cfg.TARGET_H,cfg.TARGET_W,cfg.TARGET_D), hparams=cfg.__dict__, modules=[np, tf, dk] )

    history = train(kl, cfg, gens, model_name, continuous, verbose=cfg.VEBOSE_TRAIN)

    if exp is not None:
        exp.writeAfter( model=kl.model, histories=history.history, results={"model_path": model_name} )

    try:
        notifications = read_email_config()
        notify( "Training Finished", subTitle='', message='Validation Loss {:.6f}'.format( gens.save_best.best), email_to=args['--email'], mac=True, sound=True, email_config=notifications )
    except Exception as ex:
        print( "Failed to send notifications: {}".format( ex ) )

    if cfg.SHOW_PLOT:
        fname = os.path.splitext(exp.filename)[0]
        print( "Training loss plot: {fname}_loss_acc_{loss:.6f}.png".format( fname=fname, loss=gens.save_best.best ) )
        plot_history(history, fname, gens.save_best.best, show=False)
