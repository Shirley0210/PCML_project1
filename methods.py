from costs import *
from helpers import *
from proj1_helpers import *

##########################
# Data mining functions #
##########################

# Standardization of a vector x (Function given by TAs)
def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

# Function that clean the data set, 
# i.e replace the -999 values by the column median of the valid entries
# If a collumn is entirely filled by -999, we delete it.
def clean_data(tX, medians=None):
    nbrRows = tX.shape[0]
    nbrColunms = tX.shape[1]
    
    tX_cleaned = np.zeros((nbrRows,nbrColunms))
    
    if medians is None:
        medians = np.zeros(nbrColunms)
    
    
    for columnID in range(nbrColunms):
        currentColumn = tX[:,columnID].copy()
        
        # Extract indices with -999 values
        nanIndices = []

        for rowID in range(nbrRows):
            if currentColumn[rowID] == -999.000:
                nanIndices.append(rowID)
        
        tempColumm = np.delete(currentColumn, nanIndices, axis=0)

        # Replace -999 values with median
        if medians[columnID] == 0.0:
            medians[columnID] = np.median(tempColumm)
        
        currentColumn[nanIndices] = medians[columnID]
        
        tX_cleaned[:,columnID] = currentColumn
    
    tX_reformed = tX_cleaned.copy()
    nbrOfDelete = 0
    
    for columnID in range(nbrColunms):  
        
        # Delete the column filled only by -999 values, i.e median == NaN
        if np.isnan(tX_cleaned[:, columnID]).all():
            tX_reformed = np.delete(tX_reformed, columnID - nbrOfDelete, 1)
            nbrOfDelete = nbrOfDelete + 1
    
    return tX_reformed, medians

def split_data(y, x, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    # set mask
    ratio = 0.2
    msk = np.random.rand(len(y)) < ratio
    
    # training data set
    x_tr = x[msk]
    y_tr = y[msk]
    
    # test data set
    x_test = x[~msk]
    y_test = y[~msk]
    
    return x_tr, x_test, y_tr, y_test

def cross_validation(y, tX, gamma, lambda_, max_iters, method):
    # split data
    x_tr, x_test, y_tr, y_test = split_data(y, tX)
    
    # training
    loss = 0
    weights = []
    if method == 1:
        loss, weights = least_squares_GD(y_tr, x_tr, gamma, max_iters)
    elif method == 2:
        loss, weights = least_squares_SGD(y_tr, x_tr, gamma, max_iters)
    elif method == 3:
        loss, weights = least_squares(y_tr, x_tr)
    elif method == 4:
        loss, weights = ridge_regression(y_tr, x_tr, lambda_)
    elif method == 5:
        loss, weights = logistic_regression(y_tr, x_tr, gamma, max_iters)
    else:
        loss, weights = reg_logistic_regression(y_tr, x_tr, lambda_, gamma, max_iters)
        
    # compute prediction
    y_pred = predict_labels(weights, x_test)    
    
    # accuracy of the prediction
    N = y_test.shape[0]
    pred = np.sum(y_pred == y_test)/N
        
    return pred

# Divide the data by jet features
def separating_by_jet(tX):
    
    # JET 0
    tX0_index = tX[:, 22] == 0.0
    tX0 = tX[tX0_index]
    
    # JET 1
    tX1_index = tX[:, 22] == 1.0
    tX1 = tX[tX1_index]
    
    # JET 2
    tX2_index = tX[:, 22] == 2.0
    tX2 = tX[tX2_index]
    
    # JET 3
    tX3_index = tX[:, 22] == 3.0
    tX3 = tX[tX3_index]
    
    return tX0, tX0_index, tX1, tX1_index, tX2, tX2_index, tX3, tX3_index


############################
# Machine Learning Methods #
############################



############## Linear Regressions ##############

# Gradiant calculation
def compute_gradient(y, tX, w):
    # error
    e = y - tX.dot(w)
    
    # gradient 
    N=y.shape[0]
    gradient = - np.transpose(tX).dot(e)/N
    
    return gradient


# Linear regresssion -- gradient descent
def least_squares_GD(y, tX, gamma, max_iters):
    # init parameters
    threshold = 1e-10
    w_init = np.zeros(tX.shape[1])
    ws = [w_init]
    w_temp = w_init
    losses = [8000]
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tX, w_temp)
        loss = compute_loss(y, tX, w_temp)
        
        # update w by gradient
        w_temp -= gamma*grad
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses, ws[-1]


# Stochastic gradiant calculation
def compute_stoch_gradient(y, tX, w):
    B = 35 # size of the batch (Can be change as wanted)
    sum = 0
    for minibatch_y, minibatch_tX in batch_iter(y, tX, B):
        sum += compute_gradient(minibatch_y, minibatch_tX, w)

    return sum / B


# Linear regresssion -- sctochastic gradient descent
def least_squares_SGD(y, tX, gamma, max_iters):    
    # init parameters
    threshold = 1e-8
    w_init = np.zeros(tX.shape[1])
    ws = [w_init]
    w_temp = w_init
    losses = [8000]

    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_stoch_gradient(y, tX, w_temp)
        loss = compute_loss(y, tX, w_temp)

        # update w by gradient
        w_temp -= gamma*grad
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
    # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return losses, ws[-1]


############## Least squares Regression ##############

# Least squares
def least_squares(y, tX):
    # Compute optimum weight
    tX_transpose = np.transpose(tX)
    A = tX_transpose.dot(tX)
    b = tX_transpose.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    # Compute loss
    loss = compute_loss(y, tX, w_opt)
        
    return loss, w_opt # returns loss, and optimal weights


############## Ridge Regression ##############

# Building polynomials phi, from degree 0 up to 'degree'
def build_poly(x, degree):
    poly = np.ones(x.shape)
    
    for m in range(1, degree+1):
        poly = np.c_[poly, np.power(x, m)]
   
    return poly


# Ridge regression
# The argument tX_poly represent the polynomials phi of tX
def ridge_regression(y, tX_poly, lambda_):    
    # Initiation variables
    lamb_ = 2*len(y)*lambda_
    
    # Compute optimum weight
    A = np.dot(np.transpose(tX_poly), tX_poly) + lamb_*np.eye(tX_poly.shape[1])
    b = np.transpose(tX_poly).dot(y)
    w_opt = np.linalg.solve(A,b)
    
    # Compute loss
    loss = compute_loss(y, tX_poly, w_opt)
    
    return loss, w_opt # returns mse, and optimal weights


############## Logistic Regressions ##############


# Sigmoid function
def sigmoid(t):
    temp = 1 + np.exp(-t)
    return 1/(temp)


# Gradiant descent
def learning_by_gradient_descent(y, tX, w, gamma, lambda_):
    # Initiation variables
    lamb_ = 2*len(y)*lambda_
    
    # compute the loss
    N = tX.shape[0]
    l1 = tX.dot(w) + np.log(np.ones((N))+np.exp(-tX.dot(w)))
    l2 = y*(tX.dot(w))
    loss = (np.ones((1,N)).dot(l1-l2))[0]
    
    # compute the gradient
    grad = np.transpose(tX).dot(sigmoid(tX.dot(w))-y) + lamb_*w.dot(w)
    
    # update w
    w -= gamma*grad

    return loss, w


# Logistic regression
def logistic_regression(y, tX, gamma, max_iters):
    # init parameters
    threshold = 1e-8
    w_temp = np.zeros(tX.shape[1]) # initialization of the weight
    ws = [w_temp]
    losses = [8000]

    # start the logistic regression
    for iter in range(max_iters):        
        # get loss and update w.
        loss, w_temp = learning_by_gradient_descent(y, tX, w_temp, gamma, 0)
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses, ws[-1]


# Regularized logistic regression
def reg_logistic_regression(y, tX, gamma, lambda_, max_iters):
    # init parameters
    threshold = 1e-8
    w_temp = np.zeros(tX.shape[1]) # initialization of the weight
    ws = [w_temp]
    losses = [8000]

    # start the logistic regression
    for iter in range(max_iters):        
        # get loss and update w.
        loss, w_temp = learning_by_gradient_descent(y, tX, w_temp, gamma, lambda_)
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses, ws[-1]


##############################################
# Predictions and model validation functions #
##############################################
