from costs import *
from helpers import *
from proj1_helpers import *

def clean_data(tx):
    nbrRows = tx.shape[0]
    nbrColunms = tx.shape[1]
    tx_temp = np.zeros((nbrRows,nbrColunms))
    modified_columns = [False] * nbrColunms
    
    for columnID in range(nbrColunms):
        currentColumn = tx[:,columnID].copy()
        
        # extract indices with -999 values
        nanIndices = []

        for rowID in range(nbrRows):
            if currentColumn[rowID] == -999.000:
                nanIndices.append(rowID)
                modified_columns[columnID] = True
        
        tempColumm = np.delete(currentColumn, nanIndices, axis=0)

        # replace -999 values with median
        median = np.median(tempColumm)
        currentColumn[nanIndices] = median
         
        tx_temp[:,columnID] = currentColumn
        
    return tx_temp, modified_columns


def compute_gradient(y, tX, w):
    # error
    e = y - tX.dot(w)
    
    # gradient 
    N=y.shape[0]
    gradient = - np.transpose(tX).dot(e)/N
    
    return gradient


############################
# Machine Learning Methods #
############################

# Linear regresssion -- gradient descent
def least_squares_GD(y, tx, w_init, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    ws = [w_init]
    w_temp = w_init
    losses = []
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w_temp)
        loss = compute_loss(y, tx, w_temp)
        
        # update w by gradient
        w_temp -= gamma*grad
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return ws[-1], losses[-1]


# Linear regresssion -- sctochastic gradient descent
def compute_stoch_gradient(y, tx, w):
    B = 2500 # size of the batch
    sum = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, B):
        sum += compute_gradient(minibatch_y, minibatch_tx, w)

    return sum / B

def least_squares_SGD(y, tx, w_init, max_iters, gamma):    
    # init parameters
    threshold = 1e-8
    ws = [w_init]
    w_temp = w_init
    losses = []

    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_stoch_gradient(y, tx, w_temp)
        loss = compute_loss(y, tx, w_temp)

        # update w by gradient
        w_temp -= gamma*grad
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
    # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return ws[-1], losses[-1]


# Least squares
def least_squares(y, tx):
    # Compute optimum weight
    tx_transpose = np.transpose(tx)
    A = tx_transpose.dot(tx)
    b = tx_transpose.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    # Compute loss
    loss = compute_loss(y, tx, w_opt)
        
    return w_opt, loss # returns loss, and optimal weights


# Ridge regression
def build_poly(x, degree):
    poly = np.ones(x.shape)
    
    for m in range(1, degree+1):
        poly = np.c_[poly, np.power(x, m)]
   
    return poly

def ridge_regression(y, tx, lambda_):    
    # Initiation variables
    lamb_ = 2*len(y)*lambda_
    degree = 1
    tx = build_poly(tx, degree)
    y = build_poly(y, degree)
    # Compute optimum weight
    tx_transpose = np.transpose(tx)
    A = np.dot(tx_transpose,tx) + lamb_*np.eye(tx.shape[1])
    b = tx_transpose.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    print(w_opt.shape)
    # Compute loss
    loss = compute_loss(y, tx, w_opt)
    
    return w_opt, loss # returns mse, and optimal weights


# Logistic regression
def sigmoid(t):
    temp = 1+np.exp(-t)
    return 1/(temp)

def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    # compute the loss
    N = tx.shape[0]
    l1 = tx.dot(w) + np.log(np.ones((N)) + np.exp(-tx.dot(w)))
    l2 = y*(tx.dot(w))
    penalization = lambda_*np.sum(np.power(w,2))
    loss = np.sum(l1-l2) + penalization
    
    # compute the gradient
    grad = np.transpose(tx).dot(sigmoid(tx.dot(w))-y) + 2*lambda_*w
    
    # update w
    w -= gamma*grad

    return loss, w

def logistic_regression(y, tx, w_init, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    w_temp = w_init
    ws = [w_temp]
    losses = []
    
    B = 2500 # size of the batch
    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, B):
            # get loss and update w.
            loss, w_temp = learning_by_gradient_descent(minibatch_y, minibatch_tx, w_temp, gamma, 0)
        
            # store w and loss
            ws.append(np.copy(w_temp))
            losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return ws[-1], losses[-1] 


# Regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, w_init, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    w_temp = w_init
    ws = [w_temp]
    losses = []

    # start the logistic regression
    for iter in range(max_iters):        
        # get loss and update w.
        loss, w_temp = learning_by_gradient_descent(y, tx, w_temp, gamma, lambda_)
        
        # store w and loss
        ws.append(np.copy(w_temp))
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return ws[-1], losses[-1]