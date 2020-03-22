import argparse
import json
import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow.python import keras
import keras
#from donkeycar.parts.keras import KerasVAE
#from vae_model import KerasVAE
from malpi.dkwm.vae import KerasVAE
from keras.datasets import cifar10
import donkeycar as dk
#from donkeycar.train.tub_generator import generator
from vae_generator import vae_generator
from donkeycar.templates.train import collate_records, preprocessFileList
from donkeycar.utils import gather_records
from donkeycar.parts.keras import KerasLinear, KerasIMU,\
     KerasCategorical, KerasBehavioral, Keras3D_CNN,\
     KerasRNN_LSTM, KerasLatent

from malpi.notify import notify, read_email_config
from malpi import Experiment

def plot_results( history, path=None ):
    plt.figure(1)

    # loss  - vae_r_loss  - vae_kl_loss  - val_loss  - val_vae_r_loss  - val_vae_kl_loss 
    plt.subplot(211)

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    
    # summarize history for r_loss and kl_loss (validation only)
    r_loss = 'val_vae_r_loss'
    legend1 = 'R Loss'
    if r_loss not in history:
        r_loss = 'main_output_vae_r_loss'
    if 'val_steering_output_loss' in history:
        r_loss = 'val_steering_output_loss'
        legend1 = 'Steering Loss'

    kl_loss = 'val_vae_kl_loss'
    legend2 = 'KL Loss'
    if kl_loss not in history:
        kl_loss = 'main_output_vae_kl_loss'
    if 'val_throttle_output_loss' in history:
        kl_loss = 'val_throttle_output_loss'
        legend2 = 'Throttle Loss'

    plt.subplot(212)
    plt.plot(history[r_loss])
    plt.plot(history[kl_loss])
    plt.title('R and KL Losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([legend1, legend2], loc='upper right')

    if path is not None:
        plt.savefig(path + '.png')
    plt.show()

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    print( "Train: {}".format( x_train.shape ) )
    return (x_train, x_test )

def load_tubs(cfg, tub_names, kl, aux_name=None, pilot=False):

    opts = { 'cfg' : cfg}
    opts['categorical'] = False
    opts['keras_pilot'] = kl # This will be needed for generator
    opts['continuous'] = False

    gen_records = {}
    records = gather_records(cfg, tub_names, verbose=True)
    collate_records(records, gen_records, opts)

    # These options should be part of the KerasPilot class
    if type(kl.model.output) is list:
        opts['model_out_shape'] = (2, 1)
    else:
        opts['model_out_shape'] = kl.model.output.shape

    if type(kl.model.input) is list:
        opts['model_in_shape'] = (2, 1)
    else:    
        opts['model_in_shape'] = kl.model.input.shape

    opts['has_imu'] = type(kl) is KerasIMU
    opts['has_bvh'] = type(kl) is KerasBehavioral
    opts['img_out'] = type(kl) is KerasLatent
    opts['vae_out'] = type(kl) is KerasVAE

    train_gen = vae_generator(cfg, gen_records, cfg.BATCH_SIZE, isTrainSet=True, aug=False, aux=aux_name, pilot=pilot)
    val_gen = vae_generator(cfg, gen_records, cfg.BATCH_SIZE, isTrainSet=False, aug=False, aux=aux_name, pilot=pilot)

    num_train = 0
    num_val = 0

    for key, _record in gen_records.items():
        if _record['train'] == True:
            num_train += 1
        else:
            num_val += 1

    steps = num_train // cfg.BATCH_SIZE
    val_steps = num_val // cfg.BATCH_SIZE

    print( "Num/Steps: {} {}".format( num_train, steps ) )
    print( "Val/Steps: {} {}".format( num_val, val_steps ) )

    return train_gen, val_gen, steps, val_steps

def train( kl, train_gen, val_gen, train_steps, val_steps, z_dim, beta, optim, lr=None, decay=None, momentum=None, dropout=None, epochs=40, batch_size=64, aux=None, loss_weights={} ):

    optim_args = {}
    if lr is not None:
        optim_args["lr"] = lr
    if decay is not None:
        optim_args["decay"] = decay
    if (optim == "sgd") and (momentum is not None):
        optim_args["momentum"] = momentum

    if optim == "adam":
        optim = keras.optimizers.Adam(**optim_args)
    elif optim == "sgd":
        optim = keras.optimizers.SGD(**optim_args)
    elif optim == "rmsprop":
        optim = keras.optimizers.RMSprop(**optim_args)

    kl.set_optimizer(optim)
    kl.compile(**loss_weights)
    kl.model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.00001,
                                               patience=5,
                                               verbose=True,
                                               mode='auto')

    workers_count = 1
    use_multiprocessing = False

    hist = kl.model.fit_generator(
                train_gen, 
                steps_per_epoch=train_steps, 
                epochs=epochs, 
                verbose=cfg.VEBOSE_TRAIN, 
                validation_data=val_gen,
                callbacks=[early_stop], 
                validation_steps=val_steps,
                workers=workers_count,
                use_multiprocessing=use_multiprocessing)

    return hist

def check_model( z_dim, beta ):
    shapes = [(16, 16, 3), (32, 32, 3), (64, 64, 3), (128, 128, 3)]
    for input_shape in shapes:
        print( "Checking input shape: {}".format( input_shape ) )
        kl = KerasVAE(input_shape=input_shape, z_dim=z_dim, beta=beta)
        if input_shape[0] == 128:
            kl.model.summary()
            kl2 = KerasVAE(input_shape=input_shape, z_dim=z_dim, beta=beta, aux=7)
            print( "Model with 7 auxiliary outputs" )
            kl2.compile()
            kl2.model.summary()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a VAE on Cifar10.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z_dim', type=int, default=128, help='Size of the latent space.')
    parser.add_argument('--beta', type=float, default=0.001, help='VAE beta hyperparameter.')
    parser.add_argument('--aux', default=None, help='Name of the auxilliary data to use.')
    parser.add_argument('--pilot', action="store_true", default=False, help='Train a pilot (steering/throttle)')
    parser.add_argument('--check', action="store_true", default=False, help='Check model then exit')
    parser.add_argument('--plot', action="store_true", default=False, help='Plot history in exp/name then exit')
    parser.add_argument('-o', '--optimizer', default='adam', choices=["adam", "sgd", "rmsprop"], help='Optimizer')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate. None = optimizer default')
    parser.add_argument('--decay', type=float, default=None, help='Learning rate decay. None = optimizer default')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum. None = optimizer default')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout amount to use.')
    parser.add_argument('--epochs', type=int, default=40, help='Maximum number of epoch to train.')
    parser.add_argument('--pre', default=None, help='Path to pre-trained weights to load.')

    parser.add_argument('--email', default=None, help='Email address to send finished notification.')
    parser.add_argument('--name', default=None, help='Name for this experiment run.')
    parser.add_argument('--exp', default="experiments", help='Directory where experiments are saved.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cifar10', action="store_true", default=False, help='Train on Cifar10 data.')
    group.add_argument('--file', nargs='*', help='Text file with a list of tubs to train on.')

    args = parser.parse_args()

    if args.check:
        check_model( args.z_dim, args.beta )
        exit()

    if args.plot and (args.name is not None):
        histname = os.path.join( args.exp, args.name, 'histories.pickle' )
        with open( histname, 'rb' ) as f:
            hist = pickle.load( f )
        plot_results( hist )
        exit()

    if args.cifar10:
        x_train, x_val = load_cifar10_data()
        input_shape = x_train.shape[1:]
        kl = KerasVAE(input_shape=input_shape, z_dim=args.z_dim, beta=args.beta, dropout=args.dropout)
    else:
        try:
            cfg = dk.load_config()
        except FileNotFoundError:
            cfg = dk.load_config("config.py") # retry in the current directory
        dirs = preprocessFileList( args.file )
        input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)
# Code for multiple inputs: http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/
        aux_out = 0
        if args.aux is not None:
            aux_out = 7 # need to get number of aux outputs from data
        kl = KerasVAE(input_shape=input_shape, z_dim=args.z_dim, beta=args.beta, dropout=args.dropout, aux=aux_out, pilot=args.pilot)
        train_gen, val_gen, train_steps, val_steps = load_tubs(cfg, dirs, kl, aux_name=args.aux, pilot=args.pilot)

    if args.pre is not None:
        kl.load_weights( args.pre, by_name=True )

    loss_weights = {"main_weight": 0.1, "steering_weight":100.0, "throttle_weight":100.0}
    exp = None
    if args.name is not None:
        hparams = {**cfg.__dict__, **loss_weights}
        exp = Experiment( args.name, args, exp_dir=args.exp, num_samples=train_steps+val_steps, input_dim=(cfg.TARGET_H,cfg.TARGET_W,cfg.TARGET_D), hparams=hparams, modules=[np, tf, dk] )

    hist = train( kl, train_gen, val_gen, train_steps, val_steps, args.z_dim, args.beta, args.optimizer, args.lr, args.decay, args.momentum, args.dropout, args.epochs, aux=args.aux, loss_weights=loss_weights )

# loss: 5.2231 - main_output_loss: 2.9757 - steering_output_loss: 0.0160 - throttle_output_loss: 0.0050 - main_output_vae_r_loss: 2.3089 - main_output_vae_kl_loss: 0.6668 - val_loss: 9.9828 - val_main_output_loss: 3.0794 - val_steering_output_loss: 0.0621 - val_throttle_output_loss: 0.0056 - val_main_output_vae_r_loss: 2.4030 - val_main_output_vae_kl_loss: 0.6764
    loss = hist.history['val_loss'][-1]
    if exp is not None:
        exp.writeAfter( model=kl.model, histories=hist.history, saveModel=True, results={"loss": loss} )
        print( "python3 scripts/sample.py --tub=data/20190921_4.tub {}_weights.h5".format( os.path.splitext(exp.filename)[0] ) )
        print( "python3 scripts/tubplot.py" ) # TODO: Add arguments to this once tubplot is finished

    try:
        notifications = read_email_config()
        notify( "Training Finished", subTitle='', message='Validation Loss {:.6f}'.format( loss ), email_to=args.email, mac=True, sound=True, email_config=notifications )
    except Exception as ex:
        print( "Failed to send notifications: {}".format( ex ) )

    if cfg.SHOW_PLOT:
        fname = os.path.splitext(exp.filename)[0]
        print( "Training loss plot: {}.png".format( os.path.splitext(exp.filename)[0] ) )
        plot_results( hist.history, fname )
