# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ridge regression
    N = y.shape[0]
    lamb_ = 2*N*lamb

    tx_transpose = np.transpose(tx)
    inverse = np.linalg.inv(np.dot(tx_transpose,tx) + lamb_*np.eye(tx.shape[1]))
    w_opt = np.dot(np.dot(inverse,tx_transpose),y)
    
    e = y - np.dot(tx,w_opt)
    mse = np.dot(np.transpose(e),e)/2/N
    
    return w_opt, mse
