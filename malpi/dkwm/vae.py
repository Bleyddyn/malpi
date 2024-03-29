'''
A DonkeyCar pilot based on a Variational AutoEncoder.
'''

import os
import numpy as np
import keras
import json
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Dropout
from keras.layers import SpatialDropout2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import backend as K
import tensorflow as tf

#import donkeycar as dk
from donkeycar.parts.keras import KerasPilot

from sklearn.utils import shuffle
from donkeycar.parts.augment import augment_image
from donkeycar.utils import load_scaled_image_arr
# For make_generator
from donkeycar.templates.train import collate_records
from donkeycar.utils import gather_records

"""
Possible sample code for annealing a variable during training, for the C variable in Improved Beta-VAE.

rate = K.variable(0.0,name='KL_Annealing')
annealing_rate = 0.0001
def vae_loss(y_true,y_pred):
   global annealing_rate
   global rate
   xent_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
   kl_loss = -rate * K.mean(1 + output_log - K.square(output_mean) - K.exp(output_log), axis=-1)
   rate = K.tf.assign(rate,annealing_rate)
   annealing_rate *=1.05
   rate = K.tf.assign(rate,K.clip(rate,0.0,1.0))
   return xent_loss + kl_loss
"""

def sampling_log_var(args):
    z_mean, z_log_var = args
    Z_DIM = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0.,stddev=1.)
    return z_mean + z_sigma * epsilon

def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)

class KerasVAE(KerasPilot):
    """ See this for more options for autoencoders: https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
        1) Add dropout layers to the encoder and maybe the decoder?
        2) Add an L1/L2 penalty to the mean/log_var layers?
            kernel_regularizer=regularizers.l2(l2_reg)

        The original Beta-VAE paper used beta values between 5 and 250. https://openreview.net/pdf?id=Sy2fzU9gl

        Also: Understanding disentangling in β-VAE https://arxiv.org/pdf/1804.03599.pdf
          and code: https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    """

    def __init__(self, num_outputs=2, input_shape=(128, 128, 3), z_dim=32, beta=1.0, dropout=0.4, aux=0, pilot=False, reward=False, cte=False, training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_shape
        self.z_dim = z_dim
        self.beta = beta
        self.dropout = dropout
        self.aux = aux
        self.pilot = pilot
        self.reward = reward
        self.cte = cte
        self.training = training
        self.l1_reg = 0.00001
        self.optimizer = 'adam'

        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.encoder_mu_log_var = self.models[2]
        self.decoder = self.models[3]

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

        vae_x = Input(shape=self.input_dim, name='observation_input')
        if self.dropout is not None:
            vae_xd = Dropout(self.dropout, name='dropout1')(vae_x)
        else:
            vae_xd = vae_x
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], padding="same", activation=CONV_ACTIVATIONS[0], name='conv1')(vae_xd)
        if self.dropout is not None:
            vae_c1 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c1)
            drop_num += 1
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], padding="same", activation=CONV_ACTIVATIONS[1], name='conv2')(vae_c1)
        if self.dropout is not None:
            vae_c2 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c2)
            drop_num += 1
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], padding="same", activation=CONV_ACTIVATIONS[2], name='conv3')(vae_c2)
        if self.dropout is not None:
            vae_c3 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c3)
            drop_num += 1
        vae_c3a= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], padding="same", activation=CONV_ACTIVATIONS[3], name='conv4')(vae_c3)
        if self.dropout is not None:
            vae_c3a = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c3a)
            drop_num += 1
        vae_c4= Conv2D(filters = CONV_FILTERS[4], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[4], padding="same", activation=CONV_ACTIVATIONS[4], name='conv5')(vae_c3a)
        if self.dropout is not None:
            vae_c4 = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_c4)
            drop_num += 1

        vae_z_in = Flatten(name='flatten1')(vae_c4)

        vae_z_mean = Dense(self.z_dim, kernel_regularizer=regularizers.l1(self.l1_reg), name="mu")(vae_z_in)
        vae_z_log_var = Dense(self.z_dim, kernel_regularizer=regularizers.l1(self.l1_reg), name="log_var")(vae_z_in)
        vae_z_sigma = Lambda(convert_to_sigma, name='sigma')(vae_z_log_var)

        vae_z = Lambda(sampling, name='z')([vae_z_mean, vae_z_sigma])

        vae_z_input = Input(shape=(self.z_dim,), name='z_input')

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(dense_calc, name='decoder1')

        vae_z_out = Reshape((final_img,final_img,CONV_FILTERS[3]), name='decoder_reshape')
        vae_dense_model = vae_dense(vae_z)
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], padding="same", activation=CONV_T_ACTIVATIONS[0], name='convT1')
        vae_d1_model = vae_d1(vae_z_out_model)
        if self.dropout is not None:
            vae_d1_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d1_model)
            drop_num += 1
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], padding="same", activation=CONV_T_ACTIVATIONS[1], name='convT2')
        vae_d2_model = vae_d2(vae_d1_model)
        if self.dropout is not None:
            vae_d2_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d2_model)
            drop_num += 1
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], padding="same", activation=CONV_T_ACTIVATIONS[2], name='convT3')
        vae_d3_model = vae_d3(vae_d2_model)
        if self.dropout is not None:
            vae_d3_model = SpatialDropout2D(self.dropout, name=drop_name.format(drop_num))(vae_d3_model)
            drop_num += 1
        vae_d3a = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], padding="same", activation=CONV_T_ACTIVATIONS[3], name='convT4')
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

        outputs = [vae_d4_model]

        #### Pilot outputs (e.g. Steering and Throttle)
        if self.pilot:
            if self.training:
                # During training we use samples from the mean/var distribution
                pilot_dense1 = Dense(100, name="pilot1_z")(vae_z)
                #pilot_dense1 = Dense(100, name="pilot1_flat")(vae_z_in)
            else:
                # At runtime we use just the mean
                pilot_dense1 = Dense(100, name="pilot1_mean")(vae_z_mean)
            pilot_dense2 = Dense(50, name="pilot2")(pilot_dense1)
            pilot_out = Dense(1, name="steering_output")(pilot_dense2)
            outputs.append(pilot_out)
            pilot_out = Dense(1, name="throttle_output")(pilot_dense2)
            outputs.append(pilot_out)

        if self.reward:
            reward_dense1 = Dense(100, name="reward1_z")(vae_z)
            reward_dense2 = Dense(50, name="reward2")(reward_dense1)
            reward_out = Dense(1, name="reward_output")(reward_dense2)
            outputs.append(reward_out)

        if self.cte:
            cte_dense1 = Dense(100, name="cte1_z")(vae_z)
            cte_dense2 = Dense(50, name="cte2")(cte_dense1)
            cte_out = Dense(1, name="cte_output")(cte_dense2)
            outputs.append(cte_out)

        #### Auxiliary output
        if self.aux > 0:
            aux_dense1 = Dense(100, name="aux1")(vae_z)
            aux_dense2 = Dense(50, name="aux2")(aux_dense1)
            aux_out = Dense(self.aux, name="aux_output", activation='softmax')(aux_dense2)
            outputs.append(aux_out)

        #### MODELS

        vae_full = Model(inputs=[vae_x], outputs=outputs)
        vae_encoder = Model(vae_x, vae_z)
        vae_encoder_mu_log_var = Model(vae_x, (vae_z_mean, vae_z_log_var))
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        def vae_r_loss_orig(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            return keras.losses.mean_squared_error(y_true_flat, y_pred_flat)
            #return K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)

        def vae_r_loss(y_true, y_pred):
            return tf.compat.v1.losses.mean_pairwise_squared_error(y_true, y_pred)
            #return tf.losses.mean_pairwise_squared_error(y_true, y_pred)

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + self.beta * vae_kl_loss(y_true, y_pred)
        
        self.r_loss = vae_r_loss
        self.kl_loss = vae_kl_loss
        self.loss = vae_loss

        return (vae_full, vae_encoder, vae_encoder_mu_log_var, vae_decoder)

    def set_optimizer(self, optim):
        self.optimizer = optim

    def compile(self, main_weight=1.0, steering_weight=1.0, throttle_weight=1.0, aux_weight=1.0, reward_weight=1.0, cte_weight=1.0):
        # See: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
#model.fit({'main_input': headline_data, 'aux_input': additional_data},
#          {'main_output': labels, 'aux_output': labels},
#          epochs=50, batch_size=32)
        losses={'main_output': self.loss}
        loss_weights={'main_output': main_weight}
        metrics={'main_output': [self.r_loss, self.kl_loss]}

        if self.pilot:
            losses["steering_output"] = 'mean_squared_error'
            loss_weights['steering_output'] = steering_weight
            losses["throttle_output"] = 'mean_squared_error'
            loss_weights["throttle_output"] = throttle_weight

        if self.aux > 0:
            losses['aux_output'] = 'categorical_crossentropy'
            loss_weights['aux_output'] = aux_weight
            metrics['aux_output'] = 'accuracy'

        if self.reward:
            losses['reward_output'] = 'mean_squared_error'
            loss_weights['reward_output'] = reward_weight

        if self.cte:
            losses['cte_output'] = 'mean_squared_error'
            loss_weights['cte_output'] = cte_weight

        self.model.compile(optimizer=self.optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)

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

    def save_model(self, filepath):
        """ Save the model structure to a file as json.
        """
        jstr = self.model.to_json()
        parsed = json.loads(jstr)
        arch_pretty = json.dumps(parsed, indent=4, sort_keys=True)
        with open(filepath, 'w') as f:
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

    def get_z_dim(self):
        return self.z_dim

    def decode(self, z_inputs):
        return self.decoder.predict( z_inputs )

    def encode(self, input_images):
        return self.encoder.predict( input_images )

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        (recon, steering, throttle) = self.model.predict(img_arr)
        return steering[0][0], throttle[0][0]

    def run_threaded(self, img_arr=None):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        (recon, steering, throttle) = self.model.predict(img_arr)
        return steering[0][0], throttle[0][0]

    @staticmethod
    def model_meta( fname ):
        """ Read model meta info from a json file.
            Return values will be None if not present or not used.
            @return z_dim, dropout, aux """
        aux = None
        z_dim = None
        dropout = None
        try:
            with open(fname,'r') as f:
                json_str = f.read()
                data = json.loads(json_str)
                layers = data["config"]["layers"]
                for l in layers:
                    if l.get("name","") == "aux_output":
                        aux = l["config"]["units"]
                    elif l.get("name","") == "mu":
                        z_dim = l["config"]["units"]
                    elif l.get("name","").startswith("SpatialDropout_"):
                        # e.g. SpatialDropout_0.4_1
                        dropout = float( l.get("name","").split("_")[1])
                    elif l.get("name","") == "observation_input":
                        input_shape = l["config"]["batch_input_shape"] # [null, 64, 64, 3]
                        input_shape = tuple(input_shape[1:])
        except:
            pass

        return z_dim, dropout, aux, input_shape

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

if __name__ == "__main__":
    vae = KerasVAE(z_dim=512, aux=7, pilot=True, dropout=None)
    vae.model.summary()
