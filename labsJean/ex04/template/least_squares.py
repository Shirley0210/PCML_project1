# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # least squares
    # returns mse, and optimal weights

    # Initiation variables
    mse = 0 # mean square error
    w_opt = [] # optimal weight
    N = y.shape[0]
    
    tx_transpose = np.transpose(tx)
    invert = np.linalg.inv(np.dot(tx_transpose,tx))
    w_opt = np.dot(np.dot(invert,tx_transpose),y)
    
    e = y - np.dot(tx,w_opt)
    mse = np.dot(np.transpose(e),e)/2/N
    
    
    return w_opt, mse