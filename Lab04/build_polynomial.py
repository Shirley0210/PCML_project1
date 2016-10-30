# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    poly = np.ones(x.shape)
    
    for m in range(1, degree+1):
        poly = np.c_[poly, np.power(x, m)]
   
    return poly