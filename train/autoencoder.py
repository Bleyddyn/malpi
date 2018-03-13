from __future__ import print_function

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Lambda, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.datasets import mnist
from keras import metrics

import drive_train

def vae():
    '''This script demonstrates how to build a variational autoencoder with Keras.

     #Reference

     - Auto-Encoding Variational Bayes
       https://arxiv.org/abs/1312.6114

       From: https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    batch_size = 100
    original_dim = 784
    latent_dim = 2
    intermediate_dim = 256
    epochs = 50
    epsilon_std = 1.0


    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
    vae = Model(x, x_decoded_mean)

# Compute VAE loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
#    vae_loss = K.mean(xent_loss + kl_loss)

#vae.add_loss(vae_loss)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy] )
    vae.compile(loss=vae_loss, optimizer='rmsprop', metrics=None )
    vae.summary()


# train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))

# build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

# build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def loadData():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1 ))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    return (x_train, y_train, x_test, y_test)

def makeAEConv( input_shape ):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["val_loss", "val_binary_acc", metrics.binary_accuracy])

    return autoencoder

def makeAEConvStrides( input_shape, dropouts=None ):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(24, (5, 5), activation='relu', strides=(2,2), padding='same')(input_img)
    if dropouts is not None:
        x = Dropout(dropouts[0])(x)
    x = Conv2D(32, (5, 5), activation='relu', strides=(2,2), padding='same')(x)
    if dropouts is not None:
        x = Dropout(dropouts[1])(x)
    x = Conv2D(64, (5, 5), activation='relu', strides=(2,2), padding='same')(x)
    if dropouts is not None:
        x = Dropout(dropouts[2])(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=(2,2), padding='same')(x)
    if dropouts is not None:
        x = Dropout(dropouts[3])(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same')(x)
    encoded = x

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Cropping2D(4)(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])

    return autoencoder

def train(input_dim, x_train, x_test, early_stop = False):
    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=True, mode='auto'))
    dropouts = None # [0.6,0.5,0.4,0.3]
    autoencoder = makeAEConvStrides(input_dim, dropouts = dropouts)
    autoencoder.summary()

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=callbacks)

    #autoencoder.save( 'ae_conv_model.h5' )
    json_string = autoencoder.to_json()
    with open('ae_conv.json','w') as f:
        f.write(json_string)

    autoencoder.save_weights('ae_conv_model_weights.h5')

    decoded_imgs = autoencoder.predict(x_test)

def loadModel(fname):
    model = load_model(fname)
    return model
        
def plot(input_dim, x_test):
    autoencoder = makeAEConvStrides(input_dim)
    autoencoder.load_weights('ae_conv_model_weights.h5')
    autoencoder.summary()
    decoded_imgs = autoencoder.predict(x_test, batch_size=128)
    img1 = decoded_imgs[0]
    print( "decoded: {}".format( img1.shape ) )

    n = 10
    indexes = range(len(x_test))
    random.shuffle(indexes)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        idx = indexes[i]
        img1 = x_test[idx]
        print( "test: min/max/avg {}/{}/{}".format( np.min(img1), np.max(img1), np.mean(img1) ) )
        img1 = decoded_imgs[idx]
        print( "deco: min/max/avg {}/{}/{}\n".format( np.min(img1), np.max(img1), np.mean(img1) ) )
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[idx].reshape(input_dim[0], input_dim[1], input_dim[2]))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(decoded_imgs[idx].reshape(input_dim[0], input_dim[1], input_dim[2]))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig( 'ae_conv.png')
    plt.show()

if __name__ == "__main__":
    drive_train.setCPUCores(4)

    args = drive_train.getOptions()
    images, y = drive_train.loadData(args.dirs, size=(128,128), image_norm=False)
    images = images.astype('float32') / 255.
    input_dim = images[0].shape
    #autoencoder = makeAEConvStrides(input_dim)
    #autoencoder.summary()
    #exit()

    x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=1)

    train( input_dim, x_train, x_test, early_stop=args.early)
    plot( input_dim, x_test)
