from __future__ import division

import numpy as np
import time
import cPickle

import ConfigParser
import argparse
import os.path
from os import mkdir
import sys

from load_data import load_data
from SBN import SBN

# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path with config file and possible parameters", type=str)
parser.add_argument("-r", "--restart", help="restart training with new weights", action="store_true")
parser.add_argument("-t", "--test", help="get score on test set", action="store_true")
args = parser.parse_args()

# parse the supplied config file
config = ConfigParser.ConfigParser()
config.read(args.path + "/parameters.cfg")
batch_size = config.getint('parameters', 'batch_size')
layers = config.get('parameters', 'layers')
max_epoch = config.getint('parameters', 'max_epoch')
n_samples = config.getint('parameters', 'samples')
b1 = config.getfloat('parameters', 'b1')
b2 = config.getfloat('parameters', 'b2')
learning_rate = config.getfloat('parameters', 'learning_rate')
lam = config.getfloat('parameters', 'lam')

data_train, data_valid, data_test = load_data()
layers = [data_train.shape[1]] + map(int, layers.split(','))

config.write((sys.stdout))

if args.test:
    n_samples = 1000

SBN = SBN(layers, batch_size, b1, b2, lam)

# deprecated
# SBN = herd2SBN(n_samples, layers, batch_size, b1, b2, lam, data_train.shape[0], data_valid.shape[0])

train_likelihood_list = np.array([])
valid_likelihood_list = np.array([])

epoch = 0

# check if we can continue with old weights
if not os.path.isfile(args.path + "/parameters.pkl") or args.restart:
    print "not using any saved parameters"
else:
    SBN.prng = cPickle.load(open(args.path + "/random.pkl", "rb"))
    SBN.reload(args.path)

    train_likelihood_list = np.load(args.path + "/ll_train.npy")
    valid_likelihood_list = np.load(args.path + "/ll_valid.npy")

    epoch = train_likelihood_list.shape[0]

    print "restarting at iteration: ", epoch + 1

# compile functions with initialized or loaded parameters
SBN.compile_functions(data_train, data_valid, data_test, n_samples)

if args.test:
    print "obtaining test set score with %i proposal samples" % n_samples
    test_likelihood = SBN.test()
    print "the stochastic lower bound on the log-likelihood of the test set is: ", test_likelihood

    raise SystemExit

while epoch < max_epoch:
    epoch += 1
    start = time.time()

    train_likelihood = SBN.train(learning_rate, epoch)
    steps = np.floor(data_train.shape[0] / batch_size) * epoch
    print "E: {}, S: {:.2e}, L: {}, T: {}".format(epoch, steps, train_likelihood, time.time() - start)
    train_likelihood_list = np.append(train_likelihood_list, train_likelihood)

    if epoch % 5 == 0 or epoch == 1:
        valid_likelihood = SBN.valid()
        print "the likelihood on the validation set is: ", valid_likelihood
        valid_likelihood_list = np.append(valid_likelihood_list, valid_likelihood)

    SBN.save(args.path)
    np.save(args.path + "/ll_train.npy", train_likelihood_list)
    np.save(args.path + "/ll_valid.npy", valid_likelihood_list)
    cPickle.dump(SBN.prng, open(args.path + "/random.pkl", "wb"))




