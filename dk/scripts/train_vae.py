import argparse
import json
import datetime

import matplotlib.pyplot as plt
import numpy as np
import keras
#from donkeycar.parts.keras import KerasVAE
from vae_model import KerasVAE
from keras.datasets import cifar10
import donkeycar as dk
#from donkeycar.train.tub_generator import generator
from vae_generator import generator
from donkeycar.templates.train import collate_records, preprocessFileList
from donkeycar.utils import gather_records


def plot_results( history, path ):
    plt.figure(1)

    # loss  - vae_r_loss  - vae_kl_loss  - val_loss  - val_vae_r_loss  - val_vae_kl_loss 
    plt.subplot(211)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    
    # summarize history for r_loss and kl_loss (validation only)
    r_loss = 'val_vae_r_loss'
    if r_loss not in history.history:
        r_loss = 'main_output_vae_r_loss'
    kl_loss = 'val_vae_kl_loss'
    if kl_loss not in history.history:
        kl_loss = 'main_output_vae_kl_loss'
    plt.subplot(212)
    plt.plot(history.history[r_loss])
    plt.plot(history.history[kl_loss])
    plt.title('R and KL Losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['R Loss', 'KL Loss'], loc='upper right')

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

def load_tubs(cfg, tub_names, kl, aux_name=None):

    opts = { 'cfg' : cfg}
    opts['categorical'] = False
    opts['keras_pilot'] = kl # This will be needed for generator
    opts['continuous'] = False

    gen_records = {}
    records = gather_records(cfg, tub_names, verbose=True)
    collate_records(records, gen_records, opts)

    train_gen = generator(None, opts, gen_records, cfg.BATCH_SIZE, isTrainSet=True, aug=False, aux=aux_name)
    val_gen = generator(None, opts, gen_records, cfg.BATCH_SIZE, isTrainSet=False, aug=False, aux=aux_name)

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

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for i in range(steps):
        data = next(train_gen)
        x_train.extend( data[0] )
        if aux_name is not None:
            y_train.extend( data[2] )
    for i in range(val_steps):
        data = next(val_gen)
        x_val.extend( data[0] )
        if aux_name is not None:
            y_val.extend( data[2] )

    x_train = np.array(x_train).reshape( steps*cfg.BATCH_SIZE, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH )
    x_val = np.array(x_val).reshape( val_steps*cfg.BATCH_SIZE, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH )
    if len(y_val) > 0:
        y_train = np.array(y_train).reshape( steps*cfg.BATCH_SIZE, 1 ) # hard coded for lane data
        y_val = np.array(y_val).reshape( val_steps*cfg.BATCH_SIZE, 1 )
        y_train = keras.utils.to_categorical(y_train, num_classes=7)
        y_val = keras.utils.to_categorical(y_val, num_classes=7)
    return x_train, x_val, y_train, y_val

def train( kl, x_train, x_val, z_dim, beta, optim, lr=None, decay=None, momentum=None, dropout=None, epochs=40, batch_size=64, aux=None, y_train=None, y_val=None ):

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
    kl.compile()

    kl.model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.00001,
                                               patience=5,
                                               verbose=True,
                                               mode='auto')

    if aux is not None:
        hist = kl.model.fit( x_train, {'main_output': x_train, 'aux_output': y_train},
            batch_size=batch_size, epochs=epochs, validation_data=(x_val, {'main_output': x_val, 'aux_output': y_val}), shuffle=True, callbacks=[early_stop])
    else:
        hist = kl.model.fit( x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, x_val), shuffle=True, callbacks=[early_stop])
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
    parser.add_argument('--check', action="store_true", default=False, help='Check model only')
    parser.add_argument('-o', '--optimizer', default='adam', choices=["adam", "sgd", "rmsprop"], help='Optimizer')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate. None = optimizer default')
    parser.add_argument('--decay', type=float, default=None, help='Learning rate decay. None = optimizer default')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum. None = optimizer default')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout amount to use.')
    parser.add_argument('--epochs', type=int, default=40, help='Maximum number of epoch to train.')
    parser.add_argument('--pre', default=None, help='Path to pre-trained weights to load.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cifar10', action="store_true", default=False, help='Train on Cifar10 data.')
    group.add_argument('--file', nargs='*', help='Text file with a list of tubs to train on.')

    args = parser.parse_args()

    if args.check:
        check_model( args.z_dim, args.beta )
    else:

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
            kl = KerasVAE(input_shape=input_shape, z_dim=args.z_dim, beta=args.beta, dropout=args.dropout, aux=aux_out)
            x_train, x_val, y_train, y_val = load_tubs(cfg, dirs, kl, aux_name=args.aux)
            print( "Train: {}".format( x_train.shape ) )
            print( "  Val: {}".format( x_val.shape ) )

        if args.pre is not None:
            kl.load_weights( args.pre, by_name=True )

        hist = train( kl, x_train, x_val, args.z_dim, args.beta, args.optimizer, args.lr, args.decay, args.momentum, args.dropout, args.epochs, aux=args.aux, y_train=y_train, y_val=y_val )

        n = datetime.datetime.now()
        
        fname = "vae/vae_{}_{}".format( "cifar10" if args.cifar10 else "dk", n.strftime("%Y%m%d_%H%M%S") )
        kl.model.save_weights(fname + "_weights.h5")
        print( "Saved weights to {}_weights.h5".format( fname ) )
        jstr = kl.model.to_json()
        with open(fname+"_model.json", 'w') as f:
            parsed = json.loads(jstr)
            arch_pretty = json.dumps(parsed, indent=4, sort_keys=True)
            f.write(arch_pretty)
            print( "Saved model to {}_model.json".format( fname ) )

        out = vars(args)
        out['val_loss'] = hist.history['val_loss'][-1]
        with open(fname + "_meta.json", 'w') as f:
            json.dump( out, f)

        plot_results( hist, fname )
