import gzip
import cPickle as pickle
import theano

from os.path import expanduser

def load_data():
    # Binarized MNIST, according to Salakhutdinov, Murray (2008)
    # retrieved from http://www.capsec.org/datasets/mnist_salakhutdinov.pkl.gz

    f = gzip.open('data/mnist_salakhutdinov.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = pickle.load(f)
    f.close()

    x_train = x_train.astype(theano.config.floatX)
    x_valid = x_valid.astype(theano.config.floatX)
    x_test = x_test.astype(theano.config.floatX)

    return x_train, x_valid, x_test
