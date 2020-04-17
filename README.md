# deep-memory (not fixed)

Python 3 package for memory kernel extraction from time-series using neural networks (multilayer perceptrons). Methodology is inspired by suggestions of: 

Russo, Antonio, et al. "Deep learning as closure for irreversible processes: A data-driven generalized Langevin equation." arXiv preprint arXiv:1903.09562 (2019).

The modules include one extraction technique implemented in pytorch (install torch first!). Additionally, an extraction technique implemented from scratch with bare bone Numpy is provided (not fixed!). 

Modules for correlation function caculation are adapted from memtools: https://github.com/jandaldrop/memtools.

## installation
Simply run

    pip3 install .

to install. Please find example jupyter notebooks in the example folder for explanation.

## required libarys
matplotlib, numpy, numba, sklearn, pandas, torch, scipy.
