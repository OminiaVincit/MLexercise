## Machine Learning Online Class - Exercise 1: Linear Regression

# Instructions
# ------------
# 
# This file contains code for utils function in ex1

ComputeCost <- function(X, y, theta) {
  # Compute cost for linear regression
  # J = ComputeCost(X, y, theta) computes the cost of using theta as the
  # parameter for linear regression to fit the data points in X and y
  
  # Initialize some useful values
  m = length(y) # number of training examples
  
  # ====================== YOUR CODE HERE =========================
  # Instructions: Compute the cost of a particular choice of theta
  #               You should set J to the cost.
  mat <- X %*% theta - y
  J <- (t(mat) %*% mat)/(2*m)
  return(J[1,1])
  # ===============================================================
}

ComputeCostArray <- function(X, y, theta_vals0, theta_vals1) {
  # Return cost array
  # Initialize some useful values
  vals_len = length(theta_vals0) # number of training examples
  J_vals <- c(1:vals_len)*0
  
  for ( i in 1:vals_len ) {
    theta_val <- c(theta_vals0[i], theta_vals1[i])
    J_vals[i] <- ComputeCost(X, y, theta_val)
  }
  return(J_vals)
}

GradientDescent <- function(X, y, theta, alpha, num_iters) {
  # Performs gradient descent to learn theta
  #   theta = GradientDescent(X, y, theta, alpha, num_iters) updates theta by 
  #   taking num_iters gradient steps with learning rate alpha
  # function(theta, J_history) = gradientDescent(X, y, theta, alpha, num_iters)
  
  # ====================== YOUR CODE HERE ======================
  # Instructions: Perform a single gradient step on the parameter vector theta. 
  #
  # Hint: While debugging, it can be useful to print out the values
  #       of the cost function (computeCost) and gradient here.
  #
  # Initialize some useful values
  m = length(y) # number of training examples
  n = length(theta) # number of variables theta
  J_history = c(1:num_iters) * 0
  for (iter in 1:num_iters) {
    
#     tha <- t(theta)
#     for (j in 1:n) {
#       err <- 0
#       for (i in 1:m) {
#         err <- err + (tha %*% X[i,] - y[i]) * X[i,j]
#       }
#       diff[j] <- err / m
#     }
#     # Update theta
#     theta <- theta - alpha * diff
    tx <- t(X)
    xx <- tx %*% X
    xy <- tx %*% y
    diff <- (xx %*% theta - xy) / m
    theta <- theta - alpha * diff[,1]
    # For debugging - Save the cost J in every iteration    
    J_history[iter] <- ComputeCost(X, y, theta)
  }
  return (list(theta=theta, costs=J_history))
# ============================================================
}

FeatureNormalize <- function(X) {
  # Normalize each feature (column) of data X
  # Return normalized data and mean, std
  
  num_ft <- length(X) # Number of features
  mus <- c(1:num_ft)*0
  sigmas <- c(1:num_ft)*0
  Xn <- X
  for (i in 1:num_ft) {
    mus[i] <- mean(X[,i])
    sigmas[i] <- sd(X[,i])
    Xn[,i] <- X[,i] - mus[i]
    if (sigmas[i] != 0) {
      Xn[,i] <- Xn[,i]/sigmas[i]
    }
  }
  return(list(norm=Xn,mus=mus,sigmas=sigmas))
}

NormalEqn <- function(X,y) {
  # Return closed-form solution to linear regression
  tx <- t(X) %*% X
  ty <- t(X) %*% y
  rs <- solve(tx) %*% ty
  return (rs[,1])
}