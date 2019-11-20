import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras import losses
from keras.optimizers import Adam

import tensorflow as tf

class RNN():
    def __init__(self, z_dim, action_dim, reward_dim=1, hidden_units=256, gaussian_mixtures=5, batch_size=32, epochs=20, learning_rate=0.001, optim="Adam", z_factor=1, reward_factor=1):
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_units = hidden_units
        self.gaussian_mixtures = gaussian_mixtures
        self.learning_rate = learning_rate
        self.optim = optim
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_factor = z_factor
        self.reward_factor = reward_factor

        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]

    def _build(self):

        #### THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, self.z_dim + self.action_dim + self.reward_dim))
        lstm = LSTM(self.hidden_units, return_sequences=True, return_state = True)

        lstm_output_model, _ , _ = lstm(rnn_x)
        mdn = Dense(self.gaussian_mixtures * (3*self.z_dim) + self.reward_dim)

        mdn_model = mdn(lstm_output_model)

        model = Model(rnn_x, mdn_model)

        #### THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(self.hidden_units,))
        state_input_c = Input(shape=(self.hidden_units,))
        lstm_output_forward , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

        mdn_forward = mdn(lstm_output_forward)

        forward = Model([rnn_x] + [state_input_h, state_input_c], [mdn_forward, state_h, state_c])

        #### LOSS FUNCTIONS

        def rnn_z_loss(y_true, y_pred):

            z_true, _ = self.get_responses(y_true, self.z_dim)

            d = self.gaussian_mixtures * self.z_dim
            z_pred = y_pred[:,:,:(3*d)]
            z_pred = K.reshape(z_pred, [-1, self.gaussian_mixtures * 3])

            log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

            flat_z_true = K.reshape(z_true,[-1, 1])

            z_loss = log_pi + self.tf_lognormal(flat_z_true, mu, log_sigma)
            z_loss = -K.log(K.sum(K.exp(z_loss), 1, keepdims=True))

            z_loss = K.mean(z_loss)

            return z_loss

        def rnn_rew_loss(y_true, y_pred):

            z_true, rew_true = self.get_responses(y_true, self.z_dim)

            d = self.gaussian_mixtures * self.z_dim
            reward_pred = y_pred[:,:,-1]

            rew_loss =  K.binary_crossentropy(rew_true, reward_pred, from_logits = True)

            rew_loss = K.mean(rew_loss)

            return rew_loss

        def rnn_loss(y_true, y_pred):

            z_loss = rnn_z_loss(y_true, y_pred)
            rew_loss = rnn_rew_loss(y_true, y_pred)

            return (self.z_factor * z_loss) + (self.reward_factor * rew_loss)

        opti = Adam(lr=self.learning_rate)
        model.compile(loss=rnn_loss, optimizer=opti, metrics = [rnn_z_loss, rnn_rew_loss])

        return (model,forward)

    def train(self, rnn_input, rnn_output, validation_split = 0.2):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(rnn_input, rnn_output,
            shuffle=True,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list)

        self.model.save_weights('./rnn/weights.h5')

    def train_batch(self, rnn_input, rnn_output):

        self.model.fit(rnn_input, rnn_output,
                       shuffle=False,
                       epochs=1,
                       batch_size=len(rnn_input))

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def get_responses(self, y_true, z_dim):

        z_true = y_true[:,:,:z_dim]
        rew_true = y_true[:,:,-1]

        return z_true, rew_true

    def get_mixture_coef(self, z_pred):

        log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
        log_pi = log_pi - K.log(K.sum(K.exp(log_pi), axis = 1, keepdims = True)) # axis 1 is the mixture axis

        return log_pi, mu, log_sigma

    def tf_lognormal(self, z_true, mu, log_sigma):

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((z_true - mu) / K.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI
