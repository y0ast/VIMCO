#Variational Inference for Monte Carlo objectives (VIMCO)

This repository contains all the code necessary to reproduce the results of the [VIMCO](http://arxiv.org/abs/1602.06725) paper using Theano.

Usage for CPU:

```
python main.py experiment/
```

Usage for GPU (with GPU also responsible for rendering desktop):

```
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.8 python main.py experiment/
```

Usage for GPU (GPU running headless and needs no memory for desktop):

```
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1 python main.py experiment/
```

After 2000 epochs this leads to a variational lower bound of -91.33 for 5 samples (paper obtains -93.6).

Expected output:

```
[parameters]
learning_rate = 0.001
b1 = 0.95
b2 = 0.999
batch_size = 24
layers = 200,200,200
max_epoch = 2000
samples = 5
lam = 1

not using any saved parameters
E: 1, S: 2.08e+03, L: -166.194836549, T: 10.9810910225
the likelihood on the validation set is:  -145.713702181
E: 2, S: 4.17e+03, L: -140.226117218, T: 10.8992979527
E: 3, S: 6.25e+03, L: -133.945647934, T: 10.9269690514
E: 4, S: 8.33e+03, L: -130.410928354, T: 10.9022259712
```

E means Epoch, S means Steps (as reported in the original paper), L means Log-likelihood (paper reports negative log-likelihood, which is just a sign flip) and T means Time in seconds. ~11 seconds is the runtime on a GTX 750 gpu with an Athlon II X3 425 triple core processor, on a Titan X the performance will likely be much better.

#### Notes on implementations:
The sampling is implemented separately from forward/backward for gradients. This leads to computing the forward of the inference (q) network twice, which is not optimally efficient. Also the first activation is computed on a replicated batch, an alternative is to just compute the first activation for the original batch and then sample for "repeat" amount of times. The result can be reshaped and used in further computation.


#### List of files:

- [VIMCO.py](VIMCO.py) contains the VIMCO specific code including the novel estimator
- [Model.py](Model.py) contains general model code, such as initialization of parameters, Adam updates and loading/saving of parameters.
- [SBN.py](SBN.py) contains train/test/valid functions and compiles the full Theano functions.
- [utils.py](utils.py): contains some useful utilities that are class independent
