from __future__ import division

import numpy as np
import theano.tensor as tt
import theano
from utils import replicate_batch, logsumexp, logsubexp, sigmoid
import time
from collections import OrderedDict

from Model import Model

# 1e-8 is too small and leads to NaNs
epsilon = 1e-6

class VIMCO(Model):
    def __init__(self, batch_size, b1, b2, lam):
        super(VIMCO, self).__init__(batch_size, b1, b2, lam)

    def compute_p(self, Z_below, Z_above, layer):
        Z_below_hat = sigmoid(tt.dot(Z_above, self.params['W_dec_' + str(layer)]) + self.params['b_dec_' + str(layer)])
        log_prob = -tt.nnet.binary_crossentropy(Z_below_hat, Z_below)

        return log_prob.sum(axis=1)

    def compute_q(self, Z_below, Z_above, layer):
        Z_above_hat = sigmoid(tt.dot(Z_below, self.params['W_enc_' + str(layer)]) + self.params['b_enc_' + str(layer)])
        log_prob = -tt.nnet.binary_crossentropy(Z_above_hat, Z_above)

        return log_prob.sum(axis=1)

    def compute_p_prior(self, Z):
        q = sigmoid(self.params['a'])
        p_z = -tt.nnet.binary_crossentropy(q.dimshuffle('x', 0), Z)

        return p_z.sum(axis=1)

    def compute_estimator(self, log_p_all, log_q_all):
        n_samples = tt.shape(log_p_all)[1]
        
        # See equation 14, for definition of I see equation 2
        f_x_h = log_p_all - log_q_all  # f_x_h: (batch_size, n_samples)
        sum_p_over_q = logsumexp(f_x_h, axis=1)  # sum_p_over_q: (batch_size, )
        L = sum_p_over_q - tt.log(n_samples) # L: (batch_size, )

        # Equation 10
        sum_min_i = logsubexp(sum_p_over_q.dimshuffle(0, 'x'), f_x_h)
        sum_min_i_normalized = sum_min_i - tt.log(n_samples - 1).astype(theano.config.floatX)

        L_h_given_h = L.dimshuffle(0, 'x') - sum_min_i_normalized  # equation (10)

        # Get gradient of log Q and scale
        part_1 = L_h_given_h * log_q_all  # equation 11, part 1

        weights = f_x_h - sum_p_over_q.dimshuffle(0, 'x')
        exp_weights = tt.exp(weights)

        part_2 = exp_weights * f_x_h

        # estimator = (warm_up_part_1 + part_2).sum() / self.batch_size
        estimator = (part_1 + part_2).sum() / self.batch_size

        gradients = tt.grad(estimator,
                            self.params.values(),
                            consider_constant=[exp_weights, L_h_given_h])

        likelihood = L.sum() / self.batch_size

        return likelihood, gradients

