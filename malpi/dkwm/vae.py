'''
A Variational AutoEncoder meant to be trained on DonkeyCar data.
First step in a WorldModel.
'''

import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Dropout
from keras.layers import SpatialDropout2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import backend as K
import tensorflow as tf

from donkeycar.parts.keras import KerasPilot

from sklearn.utils import shuffle
from donkeycar.parts.augment import augment_image
from donkeycar.utils import load_scaled_image_arr
# For make_generator
from donkeycar.templates.train import collate_records
from donkeycar.utils import gather_records


def sampling(args):
    z_mean, z_log_var = args
    Z_DIM = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE(KerasPilot):
    """ See this for more options for autoencoders: https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
        1) Add dropout layers to the encoder and maybe the decoder?
        2) Add an L1/L2 penalty to the mean/log_var layers?
            kernel_regularizer=regularizers.l2(l2_reg)

        The original Beta-VAE paper used beta values between 5 and 250. https://openreview.net/pdf?id=Sy2fzU9gl

        Also: Understanding disentangling in β-VAE https://arxiv.org/pdf/1804.03599.pdf
          and code: https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    """

    def __init__(self, num_outputs=2, input_shape=(128, 128, 3), z_dim=32, beta=1.0, dropout=0.4, aux=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_shape
        self.z_dim = z_dim
        self.beta = beta
        self.dropout = dropout
        self.aux = aux
        self.l1_reg = 0.00001

        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

    def _build(self):
        CONV_FILTERS = [32,64,64,128,128]
        CONV_KERNEL_SIZES = [3,3,3,3,3]
        CONV_STRIDES = [2,2,2,2,2]
        CONV_ACTIVATIONS = ['relu','relu','relu','relu', 'relu']

        CONV_T_FILTERS = [128,128,64,32,3]
        CONV_T_KERNEL_SIZES = [3,3,3,3,3]
        CONV_T_STRIDES = [2,2,2,2,2]
        CONV_T_ACTIVATIONS = ['relu','relu','relu', 'relu','sigmoid']

        final_img = int(self.input_dim[0] / (2**5)) # Reduce input image size by 5 conv layers
        dense_calc = int(final_img * final_img * CONV_FILTERS[3])
        drop_name = "SpatialDropout_{}".format( self.dropout )+"_{}"
        drop_num = 1

        vae_x = Input(shape=self.input_dim)
        if self.dropout is not None:
            vae_xd = Dropout(self.dropout)(vae_x)
        else:
            vae_xd = vae_x
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], padding="same", activation=CONV_ACTIVATIONS[0])(vae_xd)
        if self.dropout is not None:
            vae_c1 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c1)
            drop_num += 1
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], padding="same", activation=CONV_ACTIVATIONS[1])(vae_c1)
        if self.dropout is not None:
            vae_c2 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c2)
            drop_num += 1
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], padding="same", activation=CONV_ACTIVATIONS[2])(vae_c2)
        if self.dropout is not None:
            vae_c3 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c3)
            drop_num += 1
        vae_c3a= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], padding="same", activation=CONV_ACTIVATIONS[3])(vae_c3)
        if self.dropout is not None:
            vae_c3a = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c3a)
            drop_num += 1
        vae_c4= Conv2D(filters = CONV_FILTERS[4], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[4], padding="same", activation=CONV_ACTIVATIONS[4])(vae_c3a)
        if self.dropout is not None:
            vae_c4 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c4)
            drop_num += 1

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(self.z_dim, kernel_regularizer=regularizers.l1(self.l1_reg), name="z_mean")(vae_z_in)
        vae_z_log_var = Dense(self.z_dim, kernel_regularizer=regularizers.l1(self.l1_reg), name="z_var")(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(self.z_dim,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(dense_calc)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((final_img,final_img,CONV_FILTERS[3]))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], padding="same", activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        if self.dropout is not None:
            vae_d1_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d1_model)
            drop_num += 1
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], padding="same", activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        if self.dropout is not None:
            vae_d2_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d2_model)
            drop_num += 1
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], padding="same", activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        if self.dropout is not None:
            vae_d3_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d3_model)
            drop_num += 1
        vae_d3a = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], padding="same", activation=CONV_T_ACTIVATIONS[3])
        vae_d3a_model = vae_d3a(vae_d3_model)
        if self.dropout is not None:
            vae_d3a_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d3a_model)
            drop_num += 1
        vae_d4 = Conv2DTranspose(name="main_output", filters = CONV_T_FILTERS[4], kernel_size = CONV_T_KERNEL_SIZES[4] , strides = CONV_T_STRIDES[3], padding="same", activation=CONV_T_ACTIVATIONS[4])
        vae_d4_model = vae_d4(vae_d3a_model)
        #if self.dropout is not None:
        #    vae_d4_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d4_model)
        #    drop_num += 1

        if vae_d4.output_shape[1:] != self.input_dim:
            print( "Error! Input shape is not the same as Output shape: {} vs {}".format( self.input_dim, vae_d4.output_shape[1:] ) )

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d3a_decoder = vae_d3a(vae_d3_decoder)
        vae_d4_decoder = vae_d4(vae_d3a_decoder)

        #### Auxiliary output
        if self.aux > 0:
            aux_dense1 = Dense(100, name="aux1")(vae_z)
            aux_dense2 = Dense(50, name="aux2")(aux_dense1)
            aux_out = Dense(self.aux, name="aux_output")(aux_dense2)

        #### MODELS

        if self.aux > 0:
            vae = Model(inputs=[vae_x], outputs=[vae_d4_model, aux_out])
        else:
            vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        

        def vae_r_loss_orig(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            return keras.losses.mean_squared_error(y_true_flat, y_pred_flat)
            #return K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)

        def vae_r_loss(y_true, y_pred):
            return tf.losses.mean_pairwise_squared_error(y_true, y_pred)

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + self.beta * vae_kl_loss(y_true, y_pred)
        
        self.r_loss = vae_r_loss
        self.kl_loss = vae_kl_loss
        self.loss = vae_loss

        return (vae,vae_encoder, vae_decoder)

    def set_optimizer(self, optim):
        self.optimizer = optim

    def compile(self):
        # See: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
#model.fit({'main_input': headline_data, 'aux_input': additional_data},
#          {'main_output': labels, 'aux_output': labels},
#          epochs=50, batch_size=32)
        if self.aux > 0:
            loss={'main_output': self.loss, 'aux_output': 'binary_crossentropy'}
            loss_weights={'main_output': 1.0, 'aux_output': 0.2}
            self.model.compile(optimizer=self.optimizer, loss=loss,  loss_weights=loss_weights, metrics=[self.r_loss, self.kl_loss])
        else:
            self.model.compile(optimizer=self.optimizer, loss = self.loss,  metrics = [self.r_loss, self.kl_loss])

    def set_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name=by_name)

    def _train(self, data, validation_split = 0.2):
        EPOCHS = 1
        BATCH_SIZE = 32

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(data, data,
                shuffle=True,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=validation_split,
                callbacks=callbacks_list)
        
        self.model.save_weights('./vae/weights.h5')

    def save_weights(self, filepath):
        """ Save the model weights to the given filepath (should have a .h5 extension).
        """
        self.model.save_weights(filepath)

    def save(self, filepath):
        """ Save the model weights to the given filepath (should have a .h5 extension),
            and save the model structure as json to filepath - ext + ".json"
        """
        self.save_weights(filepath)

        jstr = self.model.to_json()
        jpath = os.path.splitext(filepath)[0] + ".json"

        with open(jpath, 'w') as f:
            parsed = json.loads(jstr)
            arch_pretty = json.dumps(parsed, indent=4, sort_keys=True)
            f.write(arch_pretty)

    def generate_rnn_data(self, obs_data, action_data):

        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):    
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)
    
    def decode(self, z_inputs):
        return self.decoder.predict( z_inputs )

    def encode(self, input_images):
        return self.encoder.predict( input_images )

def vae_generator(cfg, data, batch_size, isTrainSet=True, min_records_to_train=1000, aug=False, aux=None):
    """ Returns batches of data for training a VAE, given a dictionary of DonkeyCar inputs.
    """
        
    num_records = len(data)

    while True:

        batch_data = []

        keys = list(data.keys())
        keys = shuffle(keys)

        for key in keys:

            if not key in data:
                continue

            _record = data[key]

            if _record['train'] != isTrainSet:
                continue

            batch_data.append(_record)

            if len(batch_data) == batch_size:
                inputs_img = []
                aux_out = []

                for record in batch_data:
                    img_arr = None
                    #get image data if we don't already have it
                    if record['img_data'] is None:
                        img_arr = load_scaled_image_arr(record['image_path'], cfg)

                        if img_arr is None:
                            break
                        
                        if aug:
                            img_arr = augment_image(img_arr)

                        if cfg.CACHE_IMAGES:
                            record['img_data'] = img_arr
                    else:
                        img_arr = record['img_data']
                        
                    if aux is not None:
                        if aux in record['json_data']:
                            aux_out.append(record['json_data'][aux])
                        else:
                            print( "Missing aux data in: {}".format( record ) )
                            continue

                    if img_arr is None:
                        continue

                    inputs_img.append(img_arr)

                X = np.array(inputs_img).reshape(batch_size, cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
                y = X

                if aux is not None:
                    aux_out = keras.utils.to_categorical(aux_out, num_classes=7)
                    yield X, {'main_output': y, 'aux_output': aux_out}
                else:
                    yield X, y


                batch_data = []

def make_generator( cfg, tub_names, verbose=True ):
    opts = { 'cfg' : cfg}
    opts['categorical'] = False
    opts['continuous'] = False

    gen_records = {}
    records = gather_records(cfg, tub_names, verbose=True)
    collate_records(records, gen_records, opts)

    train_gen = vae_generator(cfg, gen_records, cfg.BATCH_SIZE, isTrainSet=True, aug=False )
    val_gen = vae_generator(cfg, gen_records, cfg.BATCH_SIZE, isTrainSet=False, aug=False )

    return train_gen, val_gen, gen_records


def make_config( image_h, image_w, image_d, train_split=0.8, batch_size=128, cache_images=True, crop_top=0, crop_bot=0, data_path=None, max_thr=0.5 ):
    """ Make a DonkeyCar config-like object if there is no config file available.

        Entries needed for:
        gather_records -> gather_tubs -> gather_tub_paths
            cfg.DATA_PATH if tub_names is None
        collate_records
            cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE, if opts['categorical']
            cfg.TRAIN_TEST_SPLIT
        vae_generator
            cfg.BATCH_SIZE
            cfg.CACHE_IMAGES (bool)
            cfg.IMAGE_H
            cfg.IMAGE_W
            cfg.IMAGE_DEPTH
            -> load_scaled_image_arr
                cfg.IMAGE_H
                cfg.IMAGE_W
                cfg.IMAGE_DEPTH
                -> normalize_and_crop
                    cfg.ROI_CROP_TOP
                    cfg.ROI_CROP_BOTTOM



    """
    from collections import namedtuple

    CFG = namedtuple('Config', ['DATA_PATH', 'MODEL_CATEGORICAL_MAX_THROTTLE_RANGE', 'TRAIN_TEST_SPLIT', 'BATCH_SIZE', 'CACHE_IMAGES', 'IMAGE_H', 'IMAGE_W', 'IMAGE_DEPTH', 'ROI_CROP_TOP', 'ROI_CROP_BOTTOM'])


    cfg = CFG(DATA_PATH=data_path, MODEL_CATEGORICAL_MAX_THROTTLE_RANGE=max_thr, TRAIN_TEST_SPLIT=train_split, BATCH_SIZE=batch_size, CACHE_IMAGES=cache_images, IMAGE_H=image_h, IMAGE_W=image_w, IMAGE_DEPTH=image_d, ROI_CROP_TOP=crop_top, ROI_CROP_BOTTOM=crop_bot)

    return cfg

def train( kl, train_gen, val_gen, train_steps, val_steps, z_dim, beta, optim="adam", lr=None, decay=None, momentum=None, dropout=None, epochs=40, batch_size=64, aux=None, verbose=True ):

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
                verbose=verbose, 
                validation_data=val_gen,
                callbacks=[early_stop], 
                validation_steps=val_steps,
                workers=workers_count,
                use_multiprocessing=use_multiprocessing)

    return hist

