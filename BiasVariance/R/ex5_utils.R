library("trust")
LinearRegCostFunction <- function(X, y, theta, lambda=0, grad_flag=FALSE){
  # LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  # regression with multiple variables
  #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  #   cost of using theta as the parameter for linear regression to fit the 
  #   data points in X and y. Returns the cost in J and the gradient in grad
  
  # Initialize some useful values
  m <- length(y) #number of training examples
  if (size(X,1)==1){
    X <- matrix(X,1,size(X,2))
  }
  # You need to return the following variables correctly 
  J <- 0
  grad <- rep(0, length(theta))
  diff <- X%*%theta - y
  
  # Calculate cost 
  J <- J + sum(diff*diff)/(2*m)
  if (lambda != 0){
    J <- J + (lambda/(2*m))*(sum(theta*theta) - theta[1]*theta[1])
  }
  if (!grad_flag){
    # If not calculate gradient
    return (list(value=J))
  }
  
  # Calculate gradient
  grad <- (t(X)%*%diff)[,1]/m
  B <- (t(X)%*%X)/m
  if (lambda != 0){
    grad[2:length(grad)] <- grad[2:length(grad)] + (lambda/m)*theta[2:length(theta)]
    B <- B + diag(c(0,rep(lambda/m,size(B,1)-1)),size(B,1))
  }
  return (list(value=J, gradient=grad, hessian=B))
}

TrainLinearReg <- function(X, y, lambda){
  # TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
  # regularization parameter lambda
  #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
  #   the dataset (X, y) and regularization parameter lambda. Returns the
  #   trained parameters theta.

  # Initialize Theta
  initial_theta <- rep(0, size(X,2))

  # Create "short hand" for the cost function to be minimized
  ObjFunc <- function(p) {
    return (LinearRegCostFunction(X, y, p, lambda, TRUE))
  }
#   # Minimize using gradient descent
#   max_lrate <- 1
#   relax_factor <- 0.7
#   num_iter <- 1000
#   theta <- GradientDescent(ObjFunc, initial_theta, max_lrate, num_iter, relax_factor)$params

  theta <- trust(ObjFunc, initial_theta, 1, 100, fterm=1e-6, mterm=1e-6, iterlim=100)$argument
  return(theta)
}

LearningCurve <- function(X, y, Xval, yval, lambda){
# LEARNINGCURVE Generates the train and cross validation set errors needed 
# to plot a learning curve
#   [error_train, error_val] = ...
#       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
#       cross validation set errors for a learning curve. In particular, 
#       it returns two vectors of the same length - error_train and 
#       error_val. Then, error_train(i) contains the training error for
#       i examples (and similarly for error_val(i)).
#
#   In this function, you will compute the train and test errors for
#   dataset sizes from 1 up to m. In practice, when working with larger
#   datasets, you might want to do this in larger intervals.
#

# Number of training examples
m <- size(X, 1)
mval <- size(Xval,1)

# You need to return these values correctly
error_train <- rep(0, m)
error_val   <- rep(0, m)

# ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return training errors in 
#               error_train and the cross validation errors in error_val. 
#               i.e., error_train(i) and 
#               error_val(i) should give you the errors
#               obtained after training on i examples.
#
# Note: You should evaluate the training error on the first i training
#       examples (i.e., X(1:i, :) and y(1:i)).
#
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
#
# Note: If you are using your cost function (linearRegCostFunction)
#       to compute the training and cross validation error, you should 
#       call the function with the lambda argument set to 0. 
#       Do note that you will still need to use lambda when running
#       the training to obtain the theta parameters.
#

# Hint: You can loop over the examples with the following:
  for (i in 1:m){
    # Compute train/cross validation errors using training examples 
    # X[1:i,] and y[1:i], storing the result in 
    # error_train[i] and error_val[i]
    theta <- TrainLinearReg(X[1:i,], y[1:i], lambda)
    diff <- X[1:i,]%*%theta - y[1:i]
    diff_val <- Xval%*%theta - yval
    
    error_train[i] <- mean(diff*diff)/2
    error_val[i] <- mean(diff_val*diff_val)/2
  }
  return(list(error_train=error_train, error_val=error_val))
}

PolyFeatures <- function(X, p){
  # POLYFEATURES Maps X (1D vector) into the p-th power
  #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
  #   maps each example into its polynomial features where
  #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
  # You need to return the following variables correctly.
  X_poly <- zeros(numel(X), p)

# ====================== YOUR CODE HERE ======================
  # Instructions: Given a vector X, return a matrix X_poly where the p-th 
#               column of X contains the values of X to the p-th power.
  X_poly[,1] <- X
  for(i in 2:p){
    X_poly[,i] <- X_poly[,i-1]*X
  }
  return(X_poly)
}

BFXFunc <- function(option, X, param){
  # BFS: perform operation with X and param
  m <- size(X,2)
  X <- matrix(X, size(X,1), size(X,2))
  if (option=="minus"){
    for(i in 1:m) X[,i] <- X[,i]-param[i]
  }
  if (option=="divide"){
    for (i in 1:m){
      if (param[i] != 0) X[,i] <- X[,i]/param[i]  
    }
  }
  return(X)
}

FeatureNormalize <- function(X){
  #   FEATURENORMALIZE Normalizes the features in X 
  #   FEATURENORMALIZE(X) returns a normalized version of X where
  #   the mean value of each feature is 0 and the standard deviation
  #   is 1. This is often a good preprocessing step to do when
  #   working with learning algorithms.
  m <- size(X,1)
  n <- size(X,2)
  X_norm <- zeros(m, n)
  mu <- rep(0, m)
  sigma <- rep(0, m)
  for (i in 1:n){
    mu[i] <- mean(X[,i])
    X_norm[,i] <- X[,i]-mu[i]
    sigma[i] <- std(X_norm[,i])
    if(sigma[i]!=0) X_norm[,i] <- X_norm[,i]/sigma[i]
  }
  return(list(norm=X_norm, mu=mu, sigma=sigma))
}

PlotFit <- function(min_x, max_x, mu, sigma, theta, p){
  # PLOTFIT Plots a learned polynomial regression fit over an existing figure.
  # Also works with linear regression.
  #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
  #   fit with power p and feature normalization (mu, sigma).
  # We plot a range slightly bigger than the min and max values to get
  # an idea of how the fit will vary outside the range of the data points
  X <- seq(min_x - 15, max_x + 25, 0.05)
  X <- matrix(X, length(X), 1)
  # Map the X values 
  X_poly <- PolyFeatures(X, p);
  X_poly <- BFXFunc("minus", X_poly, mu)
  X_poly <- BFXFunc("divide", X_poly, sigma)
  # Add ones
  X_poly <- cbind(ones(size(X, 1), 1), X_poly)

  # Plot
  lines(X, X_poly%*%theta, col="blue", lwd=2)
}

ValidationCurve <- function(X, y, Xval, yval){
  # VALIDATIONCURVE Generate the train and validation errors needed to
  # plot a validation curve that we can use to select lambda
  #   [lambda_vec, error_train, error_val] = ...
  #       VALIDATIONCURVE(X, y, Xval, yval) returns the train
  #       and validation errors (in error_train, error_val)
  #       for different values of lambda. 
  # You are given the training set (X, y) and validation set (Xval, yval).
  
  # Selected values of lambda (you should not change this)
  lambda_vec <- c(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
  
  # You need to return these variables correctly.
  error_train <- zeros(length(lambda_vec), 1)
  error_val <- zeros(length(lambda_vec), 1);

  # ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return training errors in 
  #               error_train and the validation errors in error_val. The 
  #               vector lambda_vec contains the different lambda parameters 
  #               to use for each calculation of the errors, i.e, 
  #               error_train(i), and error_val(i) should give 
  #               you the errors obtained after training with 
  #               lambda = lambda_vec(i)
  
  # Note: You can loop over lambda_vec with the following:
  for (i in 1:length(lambda_vec)){
    #           Compute train / val errors when training linear 
    #           regression with regularization parameter lambda
    #           You should store the result in error_train(i)
    #           and error_val(i)
    theta <- TrainLinearReg(X[1:i,], y[1:i], lambda_vec[i])
    diff <- X[1:i,]%*%theta - y[1:i]
    diff_val <- Xval%*%theta - yval
    
    error_train[i] <- mean(diff*diff)/2
    error_val[i] <- mean(diff_val*diff_val)/2  
  }
  return(list(lambda_vec=lambda_vec, error_train=error_train, error_val=error_val))
}
