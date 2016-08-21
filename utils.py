from __future__ import division

from collections import OrderedDict
import theano
import theano.tensor as tt
import numpy as np
import cPickle

epsilon = 1e-6

"""
Some useful utilities

"""

def load_parameters(path):
    theano_parameters = OrderedDict()
    saved_parameters = cPickle.load(open(path, "rb"))

    for key in saved_parameters.keys():
        cast_parameter = saved_parameters[key].astype(theano.config.floatX)

        theano_parameters[key] = theano.shared(cast_parameter, name=key)

    return theano_parameters


def save_parameters(path, parameters):
        cPickle.dump({name: p.get_value() for name, p in parameters.items()}, open(path, "wb"))

def create_theano_shared(array, name):
    return theano.shared(array.astype(theano.config.floatX), name=name)


def theano_zeros_like(array, name):
    new_array = np.zeros_like(array.get_value())
    return create_theano_shared(new_array, name)


def sigmoid(A):
    return tt.nnet.sigmoid(A) * 0.9999 + 0.000005


def logsubexp(A, B):
    """
    Numerically stable log(exp(A) - exp(B))

    """
    
    # Just adding an epsilon here does not work: the optimizer moves it out
    result = A + tt.log(1 - tt.clip(tt.exp(B - A), epsilon, 1-epsilon))
    return result

def logsumexp(A, axis=None):
    A_max = tt.max(A, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(A - A_max), axis=axis, keepdims=True)) + A_max


def replicate_batch(batch, repeat):
    batch_size, dim_data = batch.shape
    batch_ext = batch.dimshuffle((0, 'x', 1))
    batch_rep = batch_ext + tt.zeros((batch_size, repeat, dim_data), dtype=theano.config.floatX)
    batch_res = batch_rep.reshape([batch_size * repeat, dim_data])

    return batch_res

