#Variational Inference for Monte Carlo objectives (VIMCO)

This repository contains all the code necessary to reproduce the results of the [VIMCO](http://arxiv.org/abs/1602.06725) paper using Theano.

Usage:

```
python main.py experiment/
```

After 2000 epochs this leads to a variational lower bound of -91.33 for 5 samples (paper obtains -93.6).

#### Notes on implementations:
The sampling is implemented separately from forward/backward for gradients. This leads to computing the forward of the inference (q) network twice, which is not optimally efficient. Also the first activation is computed on a replicated batch, an alternative is to just compute the first activation for the original batch and then sample for "repeat" amount of times. The result can be reshaped and used in further computation.


#### List of files:

- [VIMCO.py](VIMCO.py) contains the VIMCO specific code including the novel estimator
- [Model.py](Model.py) contains general model code, such as initialization of parameters, Adam updates and loading/saving of parameters.
- [SBN.py](SBN.py) contains train/test/valid functions and compiles the full Theano functions.
- [utils.py](utils.py): contains some useful utilities that are class independent


###TO-DO

- License
- Verify results
- Add runcode for GPU
