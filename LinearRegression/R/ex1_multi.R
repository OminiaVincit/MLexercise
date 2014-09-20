## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
  # 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing learning rates).
#

## Initialization
source("utils.R")
## ================ Part 1: Feature Normalization ================

# Load Data
print('Loading data ...')
filename <- paste("../Data/ex1data2.txt")
rawData <- read.csv(filename, header=F)
X <- rawData[,1:2]
y <- rawData[,3]
m <- length(y)
n <- length(X)

# Print out some data points
print('First 10 examples from the dataset: ')
print(X[1:10,])
print(y[1:10] )
readline('Program paused. Press enter to continue.\n');

# Scale features and set them to zero mean
print('Normalizing Features ...');
        
nm_rs <- FeatureNormalize(X);
X <- nm_rs$norm
mus <- nm_rs$mus
sigmas <- nm_rs$sigmas

# Add intercept term (column 1) to X
size <- m*(n+1)
mat <- matrix(c(1:size)*0+1, nrow=m, ncol=n+1)
for (i in 1:n) {
  mat[,i+1] <- X[,i]
}
X <- mat

## ================ Part 2: Gradient Descent ================

        # ====================== YOUR CODE HERE ======================
#         % Instructions: We have provided you with the following starter
#         %               code that runs gradient descent with a particular
#         %               learning rate (alpha). 
#         %
#         %               Your task is to first make sure that your functions - 
#         %               computeCost and gradientDescent already work with 
#         %               this starter code and support multiple variables.
#         %
#         %               After that, try running gradient descent with 
#         %               different values of alpha and see which one gives
#         %               you the best result.
#         %
#         %               Finally, you should complete the code at the end
#         %               to predict the price of a 1650 sq-ft, 3 br house.
#         %
#         % Hint: By using the par command, you can plot multiple
#         %       graphs on the same figure.
#         %
#         % Hint: At prediction, make sure you do the same feature normalization.
#         %
        
  print('Running gradient descent ...');
  
  # Choose some alpha value
  alpha <- c(0.01, 0.03, 0.1, 0.3)
  num_iters <- 500
  
  # Init Theta and Run Gradient Descent 
  thetas <- list()
  J_his <- list()
  his_num <- 50
  # Plot multi figures
  old.par <- par(mfrow=c(2,2))
  for (i in 1:length(alpha)) {
    th <- c(1:(n+1))*0
    rs <- GradientDescent(X, y, th, alpha[i], num_iters)
    thetas[[i]] <- result$theta
    J_his[[i]] <- rs$costs
    title <- paste("alpha = ", alpha[i]) 
    plot(1:his_num,J_his[[i]][1:his_num], col=i+1, type="l", xlab="Num iters", ylab="Cost", main=title)
  }
  # Return to previous layout
  par(old.par)

  theta <- thetas[[3]]
  # Plot the convergence graph
  # plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
  # xlabel('Number of iterations');
  # ylabel('Cost J');
  
  # Display gradient descent's result
  print('Theta computed from gradient descent: ');
  print(theta);
  
  #Estimate the price of a 1650 sq-ft, 3 br house
  # ====================== YOUR CODE HERE ======================
    # Recall that the first column of X is all-ones. Thus, it does
  # not need to be normalized.
  price = 0  # You should change this
  
  
  # ============================================================
  print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent: ')
  nm_predict <- (c(1650,3) - mus) / sigmas
  price <- t(c(1,nm_predict)) %*% theta 
  print(price)
  readline('Program paused. Press enter to continue.\n');

  ## ================ Part 3: Normal Equations ================
  print('Solving with normal equations...')
  
  #====================== YOUR CODE HERE ======================
    # Instructions: The following code computes the closed form 
  #               solution for linear regression using the normal
  #               equations. You should complete the code in 
  #               normalEqn.R
  #
  #               After doing so, you should complete this code 
  #               to predict the price of a 1650 sq-ft, 3 br house.
  #
  theta <- NormalEqn(X, y)
  
  #Display normal equation's result
  print('Theta computed from the normal equations: ')
  print(theta)


  # Estimate the price of a 1650 sq-ft, 3 br house
  # ====================== YOUR CODE HERE ======================
  price = 0  #You should change this
  price <- t(c(1,nm_predict)) %*% theta 
  print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ')
  print(price)

