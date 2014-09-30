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

Sigmoid <- function(z){
  g <- 1/(1+exp(-z))
  return(g)
}

SigmoidGradient <- function(z){
  g <- Sigmoid(z)
  return(g*(1-g))
}

RandInitializeWeights <- function(L_in, L_out){
  # RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
  # incoming connections and L_out outgoing connections
  #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
  #   of a layer with L_in incoming connections and L_out outgoing 
  #   connections. 
  #
  #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
  #   the column row of W handles the "bias" terms
  
  # You need to return the following variables correctly 
  W <- zeros(L_out, 1 + L_in)
  epsilon_init <- sqrt(6/(L_in+L_out))
  W <- rand(L_out, 1 + L_in) *2*epsilon_init - epsilon_init
  return(W)
}

DebugInitializeWeights <- function(fan_out, fan_in){
  # DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
  # incoming connections and fan_out outgoing connections using a fixed
  # strategy, this will help you later in debugging
  #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
  #   of a layer with fan_in incoming connections and fan_out outgoing 
  #   connections using a fix set of values
  #
  #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
  #   the first row of W handles the "bias" terms
  #
  
  # Set W to zeros
  W <- zeros(fan_out, 1 + fan_in);
  
  # Initialize W using "sin", this ensures that W is always of the same
  # values and will be useful for debugging
  W <- matrix(sin(1:numel(W)), size(W,1), size(W,2)) / 10
}

NNCostFunction <- function(nn_params, input_layer_size, hidden_layer_size, 
                           num_labels, X, y, lambda, grad_flag=FALSE){
  # NNCOSTFUNCTION Implements the neural network cost function for a two layer
  # neural network which performs classification
  #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, 
  #                            X, y, lambda) 
  # computes the cost and gradient of the neural network. The
  #   parameters for the neural network are "unrolled" into the vector
  #   nn_params and need to be converted back into the weight matrices. 
  # 
  #   The returned parameter grad should be a "unrolled" vector of the
  #   partial derivatives of the neural network.
  #
  
  # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  # for our 2 layer neural network
  end <- length(nn_params)
  theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                   nrow=hidden_layer_size, ncol=(input_layer_size + 1))
  
  theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end],
                   nrow=num_labels, ncol=(hidden_layer_size + 1))
    
  # Setup some useful variables
  m <- size(X, 1)
  # You need to return the following variables correctly 
  J <- 0
  
  # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
  #               following parts.
  #
  # Part 1: Feedforward the neural network and return the cost in the
  #         variable J. After implementing Part 1, you can verify that your
  #         cost function computation is correct by verifying the cost
  #         computed in ex4.R
  #
  # Add ones to X data matrix
  
  # Forward progagation
  X <- cbind(ones(m,1), X)
  Z2 <- X%*%t(theta1)
  A2 <- 1/(1+exp(-Z2))
  A2_a <- cbind(ones(size(A2,1),1), A2) 
  Z3 <- A2_a%*%t(theta2)
  A3  <- 1/(1+exp(-Z3))
  out <- zeros(m, num_labels)
  for (i in 1:m){
    # Generate num_labels dimension output vector
    if (y[i] < 10 && y[i] > 0){
      # digit = 1:9
      out[i,y[i]] <- 1
    } 
    else{
      # digit = 0
      out[i,10] <- 1
    }
    J <- J - sum(out[i,]*log(A3[i,]) + (1-out[i,])*log(1-A3[i,]))
  }
  J <- J / m
  if( lambda != 0 ){
    J <- J + (lambda/(2*m))*(sum(theta1[,2:size(theta1,2)]^2)+
                               sum(theta2[,2:size(theta2,2)]^2))
  }
  # Return only value if don't want to calculate gradient
  if( !grad_flag ){
    return(list(value = J))
  }
  # Part 2: Implement the backpropagation algorithm to compute the gradients
  #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  #         Theta2_grad, respectively. After implementing Part 2, you can check
  #         that your implementation is correct by running checkNNGradients
  #
  #         Note: The vector y passed into the function is a vector of labels
  #               containing values from 1..K. You need to map this vector into a 
  #               binary vector of 1's and 0's to be used with the neural network
  #               cost function.
  #
  #         Hint: We recommend implementing backpropagation using a for-loop
  #               over the training examples if you are implementing it for the 
  #               first time.
  #
  theta1_grad <- zeros(size(theta1,1), size(theta1,2))
  theta2_grad <- zeros(size(theta2,1), size(theta2,2))
  
  delta3 <- A3 - out # size 5000 x 10, theta2 size: 10 x 26
  delta2 <- (delta3%*%theta2[,2:(size(theta2,2))]) * (A2*(1-A2))
  
  # For debug
#   print(size(theta1))
#   print(size(theta2))
#   print(size(X))
#   print(size(Z2))
#   print(size(A2))
#   print(size(A2_a))
#   print(size(Z3))
#   print(size(A3))
#   print(size(delta2))
#   print(size(delta3))
  
  for (i in 1:m){
    theta1_grad <- theta1_grad + (delta2[i,])%*%t(X[i,])
    theta2_grad <- theta2_grad + (delta3[i,])%*%t(A2_a[i,])
  }
  theta1_grad <- theta1_grad / m
  theta2_grad <- theta2_grad / m
  
  # Part 3: Implement regularization with the cost function and gradients.
  #
  #         Hint: You can implement this around the code for
  #               backpropagation. That is, you can compute the gradients for
  #               the regularization separately and then add them to theta1_grad
  #               and theta2_grad from Part 2.
  #
  if( lambda != 0 ){
    theta1_grad[,2:size(theta1_grad,2)] <- theta1_grad[,2:size(theta1_grad,2)] + (lambda/m)*theta1[,2:size(theta1_grad,2)]
    theta2_grad[,2:size(theta2_grad,2)] <- theta2_grad[,2:size(theta2_grad,2)] + (lambda/m)*theta2[,2:size(theta2_grad,2)]
  }
  
  # Unroll gradients
  grad <- c(theta1_grad, theta2_grad)
  return(list(value = J, gradient = grad))
}

ComputeNumericalGradient <- function(J, theta){
  # COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
  # and gives us a numerical estimate of the gradient.
  #     numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
  #     gradient of the function J around theta. Calling y = J(theta) should
  #     return the function value at theta.
  # Notes: The following code implements numerical gradient checking, and 
  #     returns the numerical gradient.It sets numgrad(i) to (a numerical 
  #     approximation of) the partial derivative of J with respect to the 
  #     i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
  #     be the (approximately) the partial derivative of J with respect to theta(i).)
  numgrad <- zeros(size(theta,1), size(theta,2))
  perturb <- zeros(size(theta,1), size(theta,2)) 
  e <- 1e-4
  for (p in 1:numel(theta)){
    # Set perturbation vector
    perturb[p] <- e
    loss1 <- J(theta - perturb)
    loss2 <- J(theta + perturb)
    # Compute Numerical Gradient
    numgrad[p] <- (loss2 - loss1) / (2*e)
    perturb[p] <- 0
  }
  return(numgrad)
}


CheckNNGradients <- function(lambda=0){
  # CHECKNNGRADIENTS Creates a small neural network to check the
  # backpropagation gradients
  #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
  #   backpropagation gradients, it will output the analytical gradients
  #   produced by your backprop code and the numerical gradients (computed
  #   using computeNumericalGradient). These two gradient computations should
  #   result in very similar values.
  
  input_layer_size <- 3
  hidden_layer_size <- 5
  num_labels <- 3
  m <- 5
  
  # We generate some 'random' test data
  theta1 <- DebugInitializeWeights(hidden_layer_size, input_layer_size)
  theta2 <- DebugInitializeWeights(num_labels, hidden_layer_size)
  
  # Reusing debugInitializeWeights to generate X
  X <- DebugInitializeWeights(m, input_layer_size - 1)
  y <- 1 + mod(1:m, num_labels)
  
  # Unroll parameters
  nn_params <- c(theta1, theta2)
  
  # Short hand for cost function
  CostFunc <- function(p){
    rs <- NNCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
    return (rs$value)
  }
  grad <- NNCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda,TRUE)$grad
  numgrad <- ComputeNumericalGradient(CostFunc, nn_params)

  # Visually examine the two gradient computations.  The two columns
  # you get should be very similar. 
  
  disp(numgrad, grad);
  print("The above two columns you get should be very similar.
          (Left-Your Numerical Gradient, Right-Analytical Gradient)")

  # Evaluate the norm of the difference between two solutions.  
  # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
  # in computeNumericalGradient.m, then diff below should be less than 1e-9
  diff <- norm(numgrad-grad)/norm(numgrad+grad)

  print("If your backpropagation implementation is correct, then 
         the relative difference will be small (less than 1e-9).
         Relative Difference: ")
  print(diff)
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