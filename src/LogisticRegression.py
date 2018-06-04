"""
This module contains the Logistic Regression functions needed for the demo files
"""

#Set output settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt



def obj(beta, X, Y, lamb):

	"""
	Computes the objective value for inputed set of model coefficients, X matrix, Y vector and regularization parameter (lamb)

    Args:
        beta: Model coefficients
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter

    Returns:
        The value of the Logistic Regression objective function.
    """
	n,d = X.shape
	return (1/n) * np.sum(np.log(1+np.exp(-Y*X.dot(beta)))) + lamb * np.sum(beta**2)

def computegrad(beta, X, Y, lamb):
	"""
	Computes the gradient for inputed set of model coefficients, X matrix, Y vector and regularization parameter (lamb)

    Args:
        beta: Model coefficients
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter

    Returns:
        The vector representing gradient of the Logistic Regression function.
    """      
	n,d = X.shape
	betax = X.dot(beta) 
	P1 = betax * -Y
	P2 = (np.exp(P1) / (1+ np.exp(P1)))
	P_mat = np.diagflat(P2)
	return -(1/n) * (X.T.dot(P_mat).dot(Y)) + 2 * lamb * beta

def backtracking(beta, X, Y, lamb, t=1, alpha=0.5, beta1=0.5, max_iter=1000):
	"""
	This is a modified version of the code from class. This function recomputes the step size t for each iteration during the training process.
    
    Args:
        beta: Model coefficients
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter

    Returns:
        Value of the step size t for the new iteration.

	"""
	grad_beta = computegrad(beta, X=X, Y=Y, lamb=lamb)
	norm_grad_x = np.linalg.norm(grad_beta)  # Norm of the gradient at x
	found_t = False
	i = 0  # Iteration counter
	while (found_t is False and i < max_iter):    
	    if (obj(beta- (t*grad_beta),X=X, Y=Y,lamb=lamb) < obj(beta, X=X, Y=Y, lamb=lamb)-alpha*t*norm_grad_x**2):
	        found_t = True
	    elif i == max_iter - 1:
	        raise Exception('Maximum number of iterations of backtracking reached')
	    else:
	        t *= beta1
	        i += 1
	return t

def my_logistic_regression(beta_init, X, Y, lamb, t_init=1 , eps = 0.001, max_iter=1000):
	"""
	Uses the algorithm for fast gradient descent, develops model by iteratively fitting training datasets   
    Args:
        beta_init: Initial starting point for beta coefficients
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter
        t_init: Intial step size, default value of 1.
        eps: Error precision, used as stopping criteria for convergence
        max_iter: Maximum number of iterations, using as stopping criteria for convergence.

    Returns:
        Vector of model coefficients from each iteration

	"""
	n,d = X.shape
	beta = beta_init
	theta = beta_init
	beta_vals = [beta]
	iter = 0
	t = t_init
	grad_theta = computegrad(theta, X=X, Y=Y, lamb=lamb)
	while np.linalg.norm(grad_theta) > eps and iter < max_iter:
	    t = backtracking(beta, X=X, Y=Y, t=t, alpha=0.5, beta1=0.5, max_iter=1000, lamb = lamb)
	    beta1 = theta - t * grad_theta
	    theta1 = beta1 + (iter/(iter+3) * (beta1-beta))
	    theta = theta1
	    beta = beta1
	    beta_vals.append(beta1)
	    grad_theta = computegrad(theta1, X=X, Y=Y, lamb=lamb)
	    iter += 1 
	return np.array(beta_vals)


def compute_t(X, lamb):
	"""
	Compute the ideal initial step size for the training the predictor set.

    Args:
        X: Predictor variable dataset (matrix)
        lamb: Lambda Regularization parameter

    Returns:
        Ideal value for initial t to use for fitting the model

	"""
	n,d = X.shape
	L = max(np.linalg.eigvals((1/n) * X.T.dot(X))) + lamb
	return 1/L

def plot_training_obj(betas, X, Y, lamb, plot_title):
	"""
	Plots the training process with iteration number and objective value.
    Args:
        betas: Model coefficients from each iteration of the training process
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter

    Returns:
        Chart showing the fitting process, minimization of the objective value

	"""
	mlr_training_obj = []
	for i in range(1,len(betas)):
	    mlr_training_obj.append(obj(betas[i,:], X=X, Y=Y, lamb=lamb))
	f, ax = plt.subplots(figsize = (8,6))
	f = plt.plot(mlr_training_obj)
	plt.title(plot_title)
	plt.xlabel("Iterations")
	plt.ylabel("Objective Values")
	plt.show();


def class_error(Y_predict, Y_actual):
    """
    	Plots the training process with iteration number and objective value.
    Args:
        Y_predict: Model prediction for classification of response variable
        Y_actual: Actual classficiation of response variable

    Returns:
        Returns float indicating the ratio of the response variables predicted correctly
    """

    prediction = []
    for i in range(0,len(Y_predict)):
        if Y_predict[i] > 0:
            prediction.append(1)
        else:
            prediction.append(-1)
    return 1-np.mean(Y_actual == prediction)


def plot_training_misclass(betas, X_train, y_train, X_test, y_test, lamb, plot_title):
	"""
	Plots the training process with iteration number and the misclassification error for the training and test set.
    Args:
        betas: Model coefficients from each iteration of the training process
        X: Predictor variable dataset (matrix)
        Y: Response variable dataset (vector)
        lamb: Lambda Regularization parameter

    Returns:
        Chart showing the fitting process, minimization of the objective value

	"""
	mlr_train_misclass = []
	mlr_test_misclass = []

	for i in range(1,len(betas)):
	    beta = betas[i,:]
	    mlr_train_misclass.append(class_error(X_train.dot(beta),y_train))
	    mlr_test_misclass.append(class_error(X_test.dot(beta),y_test))
	f, ax = plt.subplots(figsize = (8,6))
	f = plt.plot(mlr_train_misclass)
	f = plt.plot(mlr_test_misclass)
	plt.title(plot_title)
	plt.xlabel("Iterations")
	plt.ylabel("Misclassification Error")
	plt.legend(['Training Set', 'Test Set'], loc='upper right')
	plt.show();

