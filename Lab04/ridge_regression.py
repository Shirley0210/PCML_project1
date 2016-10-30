# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *

def ridge_regression(y, tX_poly, lambda_):  
	"""implement ridge regression."""
	#print(y.shape, y)
	# Initiation variables
	lamb_ = 2*len(y)*lambda_

	#tX_poly = np.zeros(tX.shape)
	#nbrRows = tX.shape[0]

	#for rowID in range(nbrRows):
	#    tX_poly[rowID, :] = build_poly(tX[rowID, :], 6)

	#tX_poly = build_poly(tX, 6)

	# Compute optimum weight
	#tX_transpose = np.transpose(tX)
	A = np.dot(np.transpose(tX_poly), tX_poly) + lamb_*np.eye(tX_poly.shape[1])
	b = np.transpose(tX_poly).dot(y)
	w_opt = np.linalg.solve(A,b)

	# Compute loss
	loss = compute_loss(y, tX_poly, w_opt)
	return loss, w_opt # returns mse, and optimal weights