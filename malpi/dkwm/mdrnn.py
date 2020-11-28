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
    def __init__(self, z_dim, action_dim, reward_dim=1, hidden_units=256, gaussian_mixtures=5, batch_size=32, epochs=20, learning_rate=0.001, optim="Adam", cte=True, z_factor=1.0, reward_factor=1.0):
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_units = hidden_units
        self.gaussian_mixtures = gaussian_mixtures
        self.learning_rate = learning_rate
        self.optim = optim
        self.epochs = epochs
        self.batch_size = batch_size
        self.do_cte = cte
        self.z_factor = z_factor
        self.reward_factor = reward_factor

        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]

    def _build(self):

        #### THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, self.z_dim + self.action_dim + self.reward_dim))
        lstm = LSTM(self.hidden_units, return_sequences=True, return_state = True, name="rnn")
        mdn = Dense(self.gaussian_mixtures * (3*self.z_dim) + self.reward_dim, name="mdn_output")
        done_hidden_layer = Dense(30, name="done_hidden")
        done_layer = Dense(1, name="done_output")

        if self.do_cte:
            cte_hidden_layer = Dense(30, name="cte_hidden")
            cte_layer = Dense(1, name="cte_output")

        lstm_output_model, _ , _ = lstm(rnn_x)
        mdn_model = mdn(lstm_output_model)
        done_model = done_layer(done_hidden_layer(lstm_output_model))
        if self.do_cte:
            cte_model = cte_layer(cte_hidden_layer(lstm_output_model))

        outputs = [mdn_model, done_model]
        if self.do_cte:
            outputs.append(cte_model)
        model = Model(rnn_x, outputs)

        #### THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(self.hidden_units,))
        state_input_c = Input(shape=(self.hidden_units,))

        lstm_output_forward , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])
        mdn_forward = mdn(lstm_output_forward)
        done_forward = done_layer(done_hidden_layer(lstm_output_forward))

        forward = Model([rnn_x] + [state_input_h, state_input_c], [mdn_forward, state_h, state_c, done_forward])

        #### LOSS FUNCTIONS

        def rnn_z_loss(y_true, y_pred):

            z_true, _ = self.get_responses(y_true, self.z_dim)

            d = self.gaussian_mixtures * self.z_dim
            z_pred = y_pred[:,:,:(3*d)]
            z_pred = K.reshape(z_pred, [-1, self.gaussian_mixtures * 3])

            log_pi, mu, log_sigma = self.get_mixture_coef_tf(z_pred)

            flat_z_true = K.reshape(z_true,[-1, 1])

            z_loss = log_pi + self.tf_lognormal(flat_z_true, mu, log_sigma)
            z_loss = -K.log(K.sum(K.exp(z_loss), 1, keepdims=True))

            z_loss = K.mean(z_loss)

            return z_loss

        def rnn_rew_loss(y_true, y_pred):

            z_true, rew_true = self.get_responses(y_true, self.z_dim)

            #d = self.gaussian_mixtures * self.z_dim
            reward_pred = y_pred[:,:,-1]

            #rew_loss =  K.binary_crossentropy(rew_true, reward_pred, from_logits = True)
            #rew_loss = K.mean(rew_loss)
            rew_loss = K.mean(K.square(reward_pred - rew_true), axis=-1)

            return rew_loss

        def rnn_loss(y_true, y_pred):

            z_loss = rnn_z_loss(y_true, y_pred)
            rew_loss = rnn_rew_loss(y_true, y_pred)

            return (self.z_factor * z_loss) + (self.reward_factor * rew_loss)

        self.z_loss = rnn_z_loss
        self.rew_loss = rnn_rew_loss
        self.loss = rnn_loss

        #opti = Adam(lr=self.learning_rate)
        #model.compile(loss=rnn_loss, optimizer=opti, metrics = [rnn_z_loss, rnn_rew_loss])

        return (model,forward)

    def compile(self, main_weight=1.0, done_weight=1.0, cte_weight=1.0):
        # See: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
#model.fit({'main_input': headline_data, 'aux_input': additional_data},
#          {'main_output': labels, 'aux_output': labels},
#          epochs=50, batch_size=32)
        losses={'mdn_output': self.loss}
        loss_weights={'mdn_output': 1.0}
        metrics={'mdn_output': [self.z_loss, self.rew_loss]}

        losses["done_output"] = 'binary_crossentropy'
        loss_weights['done_output'] = done_weight

        if self.do_cte:
            losses["cte_output"] = 'mean_squared_error'
            loss_weights['cte_output'] = cte_weight

        opti = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opti, loss=losses, loss_weights=loss_weights, metrics=metrics)

    def set_random_params(self, stdev=0.5):
        """ See: https://github.com/zacwellmer/WorldModels/blob/master/WorldModels/rnn/rnn.py#L90
        """
        params = self.model.get_weights()
        rand_params = []
        for param_i in params:
            # David Ha's initialization scheme
            sampled_param = np.random.standard_cauchy(param_i.shape)*stdev / 10000.0
            rand_params.append(sampled_param)

        self.model.set_weights(rand_params)

    def train(self, rnn_input, rnn_output, validation_split = 0.2):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        hist = self.model.fit(rnn_input, rnn_output,
            shuffle=True,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list)

        return hist

    def train_batch(self, rnn_input, rnn_output):

        self.model.fit(rnn_input, rnn_output,
                       shuffle=False,
                       epochs=1,
                       batch_size=len(rnn_input))

    def get_hidden_units(self):
        return self.hidden_units

    def set_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name=by_name)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def get_responses(self, y_true, z_dim):

        z_true = y_true[:,:,:z_dim]
        rew_true = y_true[:,:,-1]

        return z_true, rew_true

    def get_mixture_coef_tf(self, z_pred):

        log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
        log_pi = log_pi - K.log(K.sum(K.exp(log_pi), axis = 1, keepdims = True)) # axis 1 is the mixture axis

        return log_pi, mu, log_sigma

    def tf_lognormal(self, z_true, mu, log_sigma):

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((z_true - mu) / K.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI

    @staticmethod
    def get_mixture_coef_np(z_pred):
        log_pi, mu, log_sigma = np.split(z_pred, 3, 1)
        log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis = 1, keepdims = True))

        return log_pi, mu, log_sigma

    @staticmethod
    def get_pi_idx(x, pdf):
        # samples from a categorial distribution
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        random_value = np.random.randint(N)
        #print('error with sampling ensemble, returning random', random_value)
        return random_value

    @staticmethod
    def sample_z(mu, log_sigma):
        z =  mu + (np.exp(log_sigma)) * np.random.randn(*log_sigma.shape) * 0.5
        return z

    def sample_next_output(self, obs, h, c):
        
        d = self.gaussian_mixtures * self.z_dim
        
        out = self.forward.predict([np.array([[obs]]),np.array([h]),np.array([c])])
        
        y_pred = out[0][0][0]
        new_h = out[1][0]
        new_c = out[2][0]
        done = out[3][0]
        
        z_pred = y_pred[:3*d]
        rew_pred = y_pred[-1]

        z_pred = np.reshape(z_pred, [-1, self.gaussian_mixtures * 3])

        log_pi, mu, log_sigma = self.get_mixture_coef_np(z_pred)
        
        chosen_log_pi = np.zeros(self.z_dim)
        chosen_mu = np.zeros(self.z_dim)
        chosen_log_sigma = np.zeros(self.z_dim)
        
        # adjust temperatures
        pi = np.copy(log_pi)
#     pi -= pi.max()
        pi = np.exp(pi)
        pi /= pi.sum(axis=1).reshape(self.z_dim, 1)
        
        for j in range(self.z_dim):
            idx = self.get_pi_idx(np.random.rand(), pi[j])
            chosen_log_pi[j] = idx
            chosen_mu[j] = mu[j,idx]
            chosen_log_sigma[j] = log_sigma[j,idx]
            
        next_z = self.sample_z(chosen_mu, chosen_log_sigma)

        if rew_pred > 0:
            next_reward = 1
        else:
            next_reward = 0

        return next_z, chosen_mu, chosen_log_sigma, chosen_log_pi, rew_pred, next_reward, new_h, new_c, done
