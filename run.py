import numpy as np

# Functions defined in external files.
import methods as md


# Loading training data
from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Splitting the data between the 3 JETS
tX_jet0, indexes_jet0, tX_jet1, indexes_jet1, tX_jet2, indexes_jet2, tX_jet3, indexes_jet3 = md.separating_by_jet(tX)
y_jet0 = y[indexes_jet0]
y_jet1 = y[indexes_jet1]
y_jet2 = y[indexes_jet2]
y_jet3 = y[indexes_jet3]

# Cleaning Data
tX_cleaned_jet0, medians_jet0 = md.clean_data(tX_jet0)
tX_cleaned_jet1, medians_jet1  = md.clean_data(tX_jet1)
tX_cleaned_jet2, medians_jet2 = md.clean_data(tX_jet2)
tX_cleaned_jet3, medians_jet3 = md.clean_data(tX_jet3)

# We add the first column of values 1
tX_reformed_jet0 = tX_cleaned_jet0.copy()
w0 = np.ones([1,tX_reformed_jet0.shape[0]])
tX_reformed_jet0 = np.insert(tX_reformed_jet0, 0, w0, axis=1)

tX_reformed_jet1 = tX_cleaned_jet1.copy()
w0 = np.ones([1,tX_reformed_jet1.shape[0]])
tX_reformed_jet1 = np.insert(tX_reformed_jet1, 0, w0, axis=1)

tX_reformed_jet2 = tX_cleaned_jet2.copy()
w0 = np.ones([1,tX_reformed_jet2.shape[0]])
tX_reformed_jet2 = np.insert(tX_reformed_jet2, 0, w0, axis=1)

tX_reformed_jet3 = tX_cleaned_jet3.copy()
w0 = np.ones([1,tX_reformed_jet3.shape[0]])
tX_reformed_jet3 = np.insert(tX_reformed_jet3, 0, w0, axis=1)


# MODEL ON JET == 0
lambda_jet0 = 3e-10
method = 4

degree_jet0 = 3

tX_poly_jet0 = md.build_poly(tX_reformed_jet0, degree_jet0)

loss_jet0, weights_jet0 = md.ridge_regression(y_jet0, tX_poly_jet0, lambda_jet0)
pred_jet0 = md.cross_validation(y_jet0, tX_poly_jet0, 0, lambda_jet0, 0, method)



# MODEL ON JET == 1
lambda_jet1 = 3e-10
method = 4

degree_jet1 = 9

tX_poly_jet1 = md.build_poly(tX_reformed_jet1, degree_jet1)

loss_jet1, weights_jet1 = md.ridge_regression(y_jet1, tX_poly_jet1, lambda_jet1)
pred_jet1 = md.cross_validation(y_jet1, tX_poly_jet1, 0, lambda_jet1, 0, method)



# MODEL ON JET == 2
lambda_jet2 = 8e-10
method = 4

degree_jet2 = 9

tX_poly_jet2 = md.build_poly(tX_reformed_jet2, degree_jet2)

loss_jet2, weights_jet2 = md.ridge_regression(y_jet2, tX_poly_jet2, lambda_jet2)
pred_jet2 = md.cross_validation(y_jet2, tX_poly_jet2, 0, lambda_jet2, 0, method)


# MODEL ON JET == 3
lambda_jet3 = 1e-7
method = 4

degree_jet3 = 10

tX_poly_jet3 = md.build_poly(tX_reformed_jet3, degree_jet3)

loss_jet3, weights_jet3 = md.ridge_regression(y_jet3, tX_poly_jet3, lambda_jet3)
pred_jet3 = md.cross_validation(y_jet3, tX_poly_jet3, 0, lambda_jet3, 0, method)

## Making the submission file
# Loading testing data
DATA_TEST_PATH = '../data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Splitting the data between the 3 JETS
tX_test_jet0, indexes_test_jet0, tX_test_jet1, indexes_test_jet1, tX_test_jet2, indexes_test_jet2, tX_test_jet3, indexes_test_jet3 = md.separating_by_jet(tX_test)

# Cleaning Data
tX_test_cleaned_jet0, _ = md.clean_data(tX_test_jet0, medians_jet0)
tX_test_cleaned_jet1, _ = md.clean_data(tX_test_jet1, medians_jet1)
tX_test_cleaned_jet2, _ = md.clean_data(tX_test_jet2, medians_jet2)
tX_test_cleaned_jet3, _ = md.clean_data(tX_test_jet3, medians_jet3)

# Adding the first line of 1
tX_test_reformed_jet0 = tX_test_cleaned_jet0.copy()
w0 = np.ones([1,tX_test_reformed_jet0.shape[0]])
tX_test_reformed_jet0 = np.insert(tX_test_reformed_jet0, 0, w0, axis=1)

tX_test_reformed_jet1 = tX_test_cleaned_jet1.copy()
w0 = np.ones([1,tX_test_reformed_jet1.shape[0]])
tX_test_reformed_jet1 = np.insert(tX_test_reformed_jet1, 0, w0, axis=1)

tX_test_reformed_jet2 = tX_test_cleaned_jet2.copy()
w0 = np.ones([1,tX_test_reformed_jet2.shape[0]])
tX_test_reformed_jet2 = np.insert(tX_test_reformed_jet2, 0, w0, axis=1)

tX_test_reformed_jet3 = tX_test_cleaned_jet3.copy()
w0 = np.ones([1,tX_test_reformed_jet3.shape[0]])
tX_test_reformed_jet3 = np.insert(tX_test_reformed_jet3, 0, w0, axis=1)

# Building polynomials
tX_test_poly_jet0 = md.build_poly(tX_test_reformed_jet0, degree_jet0)
tX_test_poly_jet1 = md.build_poly(tX_test_reformed_jet1, degree_jet1)
tX_test_poly_jet2 = md.build_poly(tX_test_reformed_jet2, degree_jet2)
tX_test_poly_jet3 = md.build_poly(tX_test_reformed_jet3, degree_jet3)

# Making predictions
y_pred_jet0 = predict_labels(weights_jet0, tX_test_poly_jet0)
y_pred_jet1 = predict_labels(weights_jet1, tX_test_poly_jet1)
y_pred_jet2 = predict_labels(weights_jet2, tX_test_poly_jet2)
y_pred_jet3 = predict_labels(weights_jet3, tX_test_poly_jet3)

# Merging the predictions
y_pred_final = np.ones((tX_test.shape[0], 1))
a = 0
b = 0
c = 0
d = 0
for i in range(0, y_pred_final.shape[0]):
    if indexes_test_jet0[i] == True:
        y_pred_final[i] = y_pred_jet0[a]
        a = a + 1
    if indexes_test_jet1[i] == True:
        y_pred_final[i] = y_pred_jet1[b]
        b = b + 1
    if indexes_test_jet2[i] == True:
        y_pred_final[i] = y_pred_jet2[c]
        c = c + 1
    if indexes_test_jet3[i] == True:
        y_pred_final[i] = y_pred_jet3[d]
        d = d + 1
        
# Creating the submission file      
OUTPUT_PATH = '../data/dataSubmission_JET_RR.csv' 
create_csv_submission(ids_test, y_pred_final, OUTPUT_PATH)
