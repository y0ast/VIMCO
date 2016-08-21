from __future__ import division

import numpy as np
import theano
import theano.tensor as tt

from collections import OrderedDict
from utils import load_parameters, save_parameters, create_theano_shared, theano_zeros_like

epsilon = 1e-6

class Model(object):
    def __init__(self, batch_size, b1, b2, lam):
        # Adam parameters
        self.b1 = b1
        self.b2 = b2

        # Regularization parameter
        self.lam = lam
        self.t = theano.shared(0, name='t')

        self.batch_size = batch_size

        self.prng = np.random.RandomState(42)

    def compile_functions(self, data_train, data_valid, data_test, n_samples):
        self.N_train, self.N_valid, self.N_test = \
            data_train.shape[0], data_valid.shape[0], data_test.shape[0]

        self.compile_sampling(theano.shared(data_train), theano.shared(data_valid),
            theano.shared(data_test), n_samples)

        self.compile_model()

    def init_parameters(self):
        parameters = OrderedDict()

        # Prior on top
        parameters['a'] = create_theano_shared(np.zeros(self.layers[-1]), 'a')

        for i in range(len(self.layers) - 1):
            name = 'W_enc_' + str(i)
            std = 1. / np.sqrt(self.layers[i])
            value = self.prng.normal(0, std, (self.layers[i], self.layers[i + 1]))
            parameters[name] = create_theano_shared(value, name)

            name = 'b_enc_' + str(i)
            value = np.zeros(self.layers[i + 1])
            parameters[name] = create_theano_shared(value, name)

            name = 'W_dec_' + str(i)
            std = 1. / np.sqrt(self.layers[i+1])
            value = self.prng.normal(0, std, (self.layers[i+1], self.layers[i]))
            parameters[name] = create_theano_shared(value, name)

            name = 'b_dec_' + str(i)
            value = np.zeros(self.layers[i])
            parameters[name] = create_theano_shared(value, name)

        self.params = parameters

    def init_adam_parameters(self):
        m = OrderedDict()
        v = OrderedDict()

        for key, value in self.params.items():
            m[key] = theano_zeros_like(value, 'm_' + key)
            v[key] = theano_zeros_like(value, 'v_' + key)

        self.m = m
        self.v = v

    def get_batch_order(self, N):
        return range(int(N / self.batch_size))

    def get_adam_updates(self, gradients, learning_rate):
        updates = OrderedDict()
        new_t = self.t + 1
        gamma = tt.sqrt(1 - self.b2**new_t) / (1 - self.b1**new_t)

        updates[self.t] = new_t

        values_iterable = zip(self.params.keys(), self.params.values(),\
                              gradients, self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:
            new_m = self.b1 * m + (1 - self.b1) * gradient
            new_v = self.b2 * v + (1 - self.b2) * (gradient**2)

            updates[parameter] = parameter + learning_rate * gamma * \
                                 new_m / (tt.sqrt(new_v + epsilon))

            if 'W' in name or 'V' in name:
                # MAP on weights (same as L2 regularization)
                updates[parameter] -= learning_rate * self.lam *  \
                    (self.params[name] * np.float32(self.batch_size / self.N_train))

            updates[m] = new_m
            updates[v] = new_v

        return updates

    def get_sgd_updates(self, gradients, learning_rate):
        updates = OrderedDict()

        values_iterable = zip(self.params.keys(), \
                              self.params.values(), gradients)

        for name, parameter, gradient in values_iterable:
            updates[parameter] = parameter + learning_rate * gradient

        return updates

    def reload(self, path):
        self.params = load_parameters(path + "/parameters.pkl")
        self.m = load_parameters(path + "/m.pkl")
        self.v = load_parameters(path + "/v.pkl")

    def save(self, path):
        save_parameters(path + "/parameters.pkl", self.params)
        save_parameters(path + "/m.pkl", self.m)
        save_parameters(path + "/v.pkl", self.v)
