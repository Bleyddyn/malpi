import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import cifar10
import keras

from donkeycar.parts.keras import KerasVAE

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    print( "Train: {}".format( x_train.shape ) )
    return (x_train, x_test )

def sample_vae(vae, count, path):
    z_size = vae.z_dim
    batch_size=count

    z = np.random.normal(size=(batch_size,z_size))
    samples = vae.decode(z)
    input_dim = samples.shape[1:]

    n = batch_size
    plt.figure(figsize=(20, 6), tight_layout=False)
    plt.title('VAE samples')
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Random")

    x_train, x_test = load_data()
    np.random.shuffle(x_test)
    orig = x_test[0:batch_size]
    recon = vae.decode( vae.encode(orig) )

    for i in range(n):
        ax = plt.subplot(3, n, n+i+1)
        plt.imshow(orig[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Real")

        ax = plt.subplot(3, n, (2*n)+i+1)
        plt.imshow(recon[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if 0 == i:
            ax.set_title("Reconstructed")

    path = os.path.splitext(path)[0] + "_sample.png"
    plt.savefig( path )
    plt.show()

def main(model, count, z_dim):
    input_shape = (32, 32, 3)
    beta = 1.0

    kl = KerasVAE(input_shape=input_shape, z_dim=z_dim, beta=beta, dropout=None)
    kl.set_weights(model)

    #arch = model + "_model.json"
    #with open(arch, 'r') as f:
    #    json_str = f.read()
    #kl = keras.models.model_from_json( json_str )
    #weights = model + "_weights.h5"
    #kl.load_weights(weights)

    sample_vae(kl, count, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a trained VAE on Cifar10.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--count', type=int, default=10, help='How many samples to display.')
    parser.add_argument('--z_dim', type=int, default=128, help='Must be the same size as the trained model.')
    parser.add_argument('model', nargs=1, help='Saved model file')

    args = parser.parse_args()

    main(args.model[0], 10, args.z_dim)
