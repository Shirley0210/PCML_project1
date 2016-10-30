# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    #print(y.shape, tx.shape, w.shape)
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def calculate_mse(e):
    return 1 / 2 * np.mean(e ** 2)


def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)
