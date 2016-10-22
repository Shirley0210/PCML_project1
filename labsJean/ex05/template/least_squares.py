# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    
def least_squares(y, tx):
    # Initiation variables
    loss = 0 # loss
    w_opt = [] # optimal weight

    # Compute optimum weight
    tx_transpose = np.transpose(tx)
    invert = np.linalg.inv(tx_transpose.dot(tx))
    w_opt = np.dot(invert.dot(tx_transpose),y)
    
    # Compute loss
    loss = compute_loss(y, tx, w_opt)
    
    return loss, w_opt # returns loss, and optimal weights
