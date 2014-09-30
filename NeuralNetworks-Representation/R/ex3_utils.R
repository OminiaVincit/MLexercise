library("trust")
DisplayData <- function(X, example_width=-1){
  # DISPLAYDATA Display 2D data in a nice grid
  #   stored in X in a nice grid. It returns the figure handle h and the
  #   displayed array if requested
  
  # Set example_width automatically if not passed in
  if (example_width == -1){
    example_width <- round(sqrt(size(X,2)))
  }
  # Compute rows, cols
  m <- size(X,1)
  n <- size(X,2)
  example_height <- n/example_width
  # Compute number of items to display
  display_rows <- floor(sqrt(m))
  display_cols <- ceil(m/display_rows)
  
  # Between images padding
  pad <- 1
  
  # Setup blank display
  display_array <- -ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad))
  
  # Copy each example into a patch on the display array
  curr_ex <- 1
  for (j in 1:display_rows){
    for (i in 1:display_cols) {
      if (curr_ex > m) {
        break
      }
      # Copy the patch
      # Get the max value of the patch
      max_val <- max(abs(X[curr_ex, ]))
      disx <- pad+(j-1)*(example_height+pad) + (1:example_height)
      disy <- pad+(i-1)*(example_width+pad) + (example_width:1)
      display_array[disx, disy] <- t(matrix(X[curr_ex, ],nrow=example_width,ncol=example_height)) / max_val
      curr_ex <- curr_ex+1
    }
    if (curr_ex > m){
      break
    }
  }
  # Display Image
  h <- image(display_array, axes=FALSE, col=grey(seq(0,1,length=256)))
}

Sigmod <- function(theta, x) {
  z <- t(theta)%*%x
  rs <-  1/(1+exp(-z))
  return (rs[1,1])
}

LrCostFunction <- function(theta, X, y, lambda=0){
  # LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  # regularization
  #   [J, grad, hess] = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  #   theta as the parameter for regularized logistic regression and the
  #   gradient, hessian of the cost w.r.t. to the parameters. 
  
  # Initialize some useful values
  m <- size(X,1)# number of training examples
  n <- size(X,2)
  z <- (X%*%theta)[,1]
  sigs <- 1/(1+exp(-z))
  cost <- sum(-y*log(sigs) - (1-y)*log(1-sigs))/m
  grad <- (t(X)%*%(sigs-y))[,1]
  grad <- grad / m
  dg <- sigs*(1-sigs)/m
  D <- diag(sqrt(dg), n, m)%*%X
  B <- t(D)%*%D
  if (lambda != 0){
    # Regularization case
    cost <- cost + lambda*sum(theta^2)/(2*m)
    grad <- grad + lambda*c(0,theta[2:n])/m
    B <- B + diag(c(0,rep(lambda/m,n-1)),n)
  }
  return(list(value = cost, gradient = grad, hessian = B))
}

OneVsAll <- function(X, y, numlabels, lambda){
  # ONEVSALL trains multiple logistic regression classifiers and returns all
  # the classifiers in a matrix all_theta, where the i-th row of all_theta 
  # corresponds to the classifier for label i
  #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
  #   logisitc regression classifiers and returns each of these classifiers
  #   in a matrix all_theta, where the i-th row of all_theta corresponds 
  #   to the classifier for label i
  
  m <- size(X,1)# number of training examples
  n <- size(X,2)
  all_thetas <- zeros(numlabels, n+1)
  
  # Add ones to X data matrix
  X <- cbind(ones(m,1), X)
  
  # Hint: theta(:) will return a column vector.
  #
  # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
  #       whether the ground truth is true/false for this class.

  for (c in 1:numlabels){
    print(paste("c=",c))
    #  Set Initial theta
    initial_theta <- rep(0,n+1)
    ObjFunc <- function(th) {
      return (LrCostFunction(th, X, (y==c)))
    }
    whoop <- trust(ObjFunc, initial_theta, 1, 100, fterm=1e-6, mterm=1e-6, iterlim=100)
    all_thetas[c,] <- whoop$argument
  }
  return (all_thetas)
}

PredictOneVsAll <- function(all_theta, X){
  # PREDICT Predict the label for a trained one-vs-all classifier. The labels 
  # are in the range 1..K, where K = size(all_theta, 1). 
  #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
  #  for each example in the matrix X. Note that X contains the examples in
  #  rows. all_theta is a matrix where the i-th row is a trained logistic
  #  regression theta vector for the i-th class. You should set p to a vector
  #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2 for 4 examples) 
  
  m <- size(X,1)# number of training examples
  n <- size(X,2)
  p <- rep(0, n)
  
  # Add ones to X data matrix
  X <- cbind(ones(m,1), X)
  
  # Instructions: Complete the following code to make predictions using
  #               your learned logistic regression parameters (one-vs-all).
  #               You should set p to a vector of predictions (from 1 to num_labels).
  #
  # Hint: This code can be done all vectorized using the which.max function.
  #
  Z <- X%*%t(all_theta)
  Z <- 1/(1+exp(-Z))
  rsy <- rep(0,m)
  for(i in 1:m){
    rsy[i] <- which.max(Z[i,])
  }
  return (rsy)
}

Predict <- function(theta1, theta2, X){
  # PREDICT Predict the label of an input given a trained neural network
  #   p = PREDICT(theta1, theta2, X) outputs the predicted label of X given the
  #   trained weights of a neural network (theta1, theta2)
  
  # Useful values
  m <- size(X, 1)
  num_labels <- size(theta2, 1)
  
  # Add ones to X data matrix
  X <- cbind(ones(m,1), X)
  
  # You need to return the following variables correctly 
  p <- rep(0, m)
  
  # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
  #               your learned neural network. You should set p to a 
  #               vector containing labels between 1 to num_labels.
  #
  # Hint: The which.max function might come in useful. 
  Z2 <- X%*%t(theta1)
  A2 <- 1/(1+exp(-Z2))
  A2 <- cbind(ones(size(A2,1),1), A2)
  Z3 <- A2%*%t(theta2)
  A  <- 1/(1+exp(-Z3))
  rsy <- rep(0,m)
  for(i in 1:m){
    rsy[i] <- which.max(A[i,])
  }
  return (rsy)
}