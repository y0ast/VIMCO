from __future__ import division

import theano.tensor as tt
import theano
import numpy as np

from VIMCO import VIMCO

from utils import sigmoid, replicate_batch

class SBN(VIMCO):
    def __init__(self, layers, batch_size, b1, b2, lam):
        super(SBN, self).__init__(batch_size, b1, b2, lam)

        self.layers = layers

        self.init_parameters()
        self.init_adam_parameters()

    def train(self, learning_rate):
        batch_likelihood_list = np.array([])
        batch_order = self.get_batch_order(self.N_train)
        self.prng.shuffle(batch_order)

        for i, batch in enumerate(batch_order):
            samples = self.sample_train(batch)
            batch_L = self.update_train(learning_rate, *samples)

            batch_likelihood_list = np.append(batch_likelihood_list, batch_L)

        assert(batch_likelihood_list.shape[0] == len(batch_order))

        return np.mean(batch_likelihood_list)

    def valid(self):
        batch_likelihood_list = np.array([])
        batch_order = self.get_batch_order(self.N_valid)
        self.prng.shuffle(batch_order)

        for i, batch in enumerate(batch_order):
            samples = self.sample_valid(batch)
            batch_L = self.likelihood_valid(*samples)

            batch_likelihood_list = np.append(batch_likelihood_list, batch_L)

        assert(batch_likelihood_list.shape[0] == len(batch_order))

        return np.mean(batch_likelihood_list)

    def test(self):
        batch_likelihood_list = np.array([])
        batch_order = self.get_batch_order(self.N_test)
        self.prng.shuffle(batch_order)

        for i, batch in enumerate(batch_order):
            samples = self.sample_test(batch)
            batch_L = self.likelihood_test(*samples)

            batch_likelihood_list = np.append(batch_likelihood_list, batch_L)

        assert(batch_likelihood_list.shape[0] == len(batch_order))

        return np.mean(batch_likelihood_list)

    def compute_samples(self, srng, Z_below, layer):
        q_z_above_given_z_below = sigmoid(tt.dot(Z_below, self.params['W_enc_' + str(layer)]) + self.params['b_enc_' + str(layer)])

        U = srng.uniform(q_z_above_given_z_below.shape)        
        Z = tt.cast(q_z_above_given_z_below > U, dtype=theano.config.floatX)

        return Z

    def compile_sampling(self, data_train, data_valid, data_test, training_n_samples):
        X = tt.matrix('X')
        batch = tt.iscalar('batch')
        n_samples = tt.iscalar('n_samples')

        n_layers = len(self.layers)
        samples = [None] * n_layers

        samples[0] = replicate_batch(X, n_samples)

        if "gpu" in theano.config.device:
            from theano.sandbox import rng_mrg
            srng = rng_mrg.MRG_RandomStreams(seed=42)
        else:
            srng = tt.shared_randomstreams.RandomStreams(seed=42)
         
        for layer in range(n_layers - 1):
            samples[layer + 1] = self.compute_samples(srng, samples[layer], layer)


        givens = dict()
        givens[X] = data_valid[batch * self.batch_size:(batch + 1) * self.batch_size]
        self.sample_convergence = theano.function([batch, n_samples], samples, givens=givens)

        givens[n_samples] = np.int32(training_n_samples)
        givens[X] = data_train[batch * self.batch_size:(batch + 1) * self.batch_size]
        self.sample_train = theano.function([batch], samples, givens=givens)

        givens[X] = data_valid[batch * self.batch_size:(batch + 1) * self.batch_size]
        self.sample_valid = theano.function([batch], samples, givens=givens)

        givens[X] = data_test[batch * self.batch_size:(batch + 1) * self.batch_size]
        self.sample_test = theano.function([batch], samples, givens=givens)

    def compile_model(self):
        learning_rate = tt.scalar('learning_rate')

        n_layers = len(self.layers)
        samples = [tt.matrix('samples_' + str(i)) for i in range(n_layers)]
        log_q = [None] * n_layers
        log_p = [None] * n_layers

        n_samples = tt.cast(tt.shape(samples[0])[0] / self.batch_size, dtype="int32")
        log_q[0] = tt.zeros((tt.shape(samples[0])[0],), dtype=theano.config.floatX)

        for layer in range(n_layers - 1):
            log_q[layer + 1] = self.compute_q(samples[layer], 
                                              samples[layer +1],
                                              layer)

        log_p[-1] = self.compute_p_prior(samples[-1])
        for layer in range(n_layers - 1, 0, -1):
            log_p[layer - 1] = self.compute_p(samples[layer - 1],
                                              samples[layer], layer - 1)

        log_p_all = tt.zeros((self.batch_size, n_samples), dtype=theano.config.floatX)
        log_q_all = tt.zeros((self.batch_size, n_samples), dtype=theano.config.floatX)

        for layer in range(n_layers):
            log_q[layer] = log_q[layer].reshape((self.batch_size, n_samples))
            log_p[layer] = log_p[layer].reshape((self.batch_size, n_samples))

            log_p_all += log_p[layer]
            log_q_all += log_q[layer]

        likelihood, gradients = self.compute_estimator(log_p_all, log_q_all)

        updates = self.get_adam_updates(gradients, learning_rate)

        self.update_train = theano.function([learning_rate] + samples,
                                            likelihood,
                                            updates=updates)

        self.likelihood_valid = theano.function(samples, likelihood)
        self.likelihood_test = theano.function(samples, likelihood)
        self.get_gradients = theano.function(samples, gradients)

