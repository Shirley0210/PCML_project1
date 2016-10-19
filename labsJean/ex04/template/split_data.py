# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    msk = np.random.rand(len(y)) < ratio
    x_tr = x[msk]
    x_test = x[~msk]
    y_tr = y[msk]
    y_test = y[~msk]
    
    return x_tr, x_test, y_tr, y_test
