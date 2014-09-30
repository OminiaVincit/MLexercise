## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
  # 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this ex4_utils.R
  #
#     SigmoidGradient.m
#     RandInitializeWeights.m
#     NNCostFunction.m
#
## Initialization
library("R.matlab")
library("pracma")
source("ex4_utils.R")
source("optimizer.R")

## Setup the parameters you will use for this exercise
input_layer_size  <- 400 # 20x20 Input Images of Digits
hidden_layer_size <- 25  # 25 hidden units
num_labels <- 10          # 10 labels, from 1 to 10   
# (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
  #  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')
pathname <- file.path("../Data", "ex4data1.mat")
data <- readMat(pathname) # training data stored in arrays X, y
X <- data$X
y <- data$y
m <- size(X,1)
n <- size(X,2)

# Randomly select 100 data points to display
rand_indices <- randperm(m,100)
sel <- X[rand_indices, ]
DisplayData(sel)
readline('Program paused. Press enter to continue.')


## ================ Part 2: Loading Parameters ================
  # In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights (trained) into variables Theta1 and Theta2
pathname <- file.path("../Data", "ex4weights.mat")
data <- readMat(pathname)
theta1 <- data$Theta1 # size 25 x 401
theta2 <- data$Theta2 # size 10 x 26

## Unroll parameters 
nn_params <- c(theta1, theta2)

# # ================ Part 3: Compute Cost (Feedforward) ================
# #     To the neural network, you should first start by implementing the
# #   feedforward part of the neural network that returns the cost only. You
# #   should complete the code in nnCostFunction.m to return cost. After
# #   implementing the feedforward to compute the cost, you can verify that
# #   your implementation is correct by verifying that you get the same cost
# #   as us for the fixed debugging parameters.
# # 
# #   We suggest implementing the feedforward cost *without* regularization
# #   first so that it will be easier for you to debug. Later, in part 4, you
# #   will get to implement the regularized cost.
# # 
# print("Feedforward Using Neural Network ...")
# # Weight regularization parameter (we set this to 0 here).
# lambda <- 0 
# J <- NNCostFunction(nn_params, input_layer_size, hidden_layer_size, 
#                     num_labels, X, y, lambda)
# 
# print(paste("Cost at parameters (loaded from ex4weights)", 
#             J$value, " (this value should be about 0.287629)"))
# readline('Program paused. Press enter to continue.')
# 
# # 
# # =============== Part 4: Implement Regularization ===============
# #   Once your cost function implementation is correct, you should now
# #   continue to implement the regularization with the cost.
# # 
# print("Checking Cost Function (Regularization) ... ")
# 
# # Weight regularization parameter (we set this to 1 here).
# lambda <- 1.0
# J <- NNCostFunction(nn_params, input_layer_size, hidden_layer_size, 
#                     num_labels, X, y, lambda)
# print(paste("Cost at parameters (loaded from ex4weights)", 
#             J$value, " (this value should be about 0.383770)"))
# readline('Program paused. Press enter to continue.')
# 
# # 
# # ================ Part 5: Sigmoid Gradient  ================
# #     Before you start implementing the neural network, you will first
# #   implement the gradient for the sigmoid function. You should complete the
# #   code in the sigmoidGradient.m file.
# # 
# # 
# print("Evaluating sigmoid gradient...")
# g <- SigmoidGradient(c(-1,-0.5,0,0.5,1))
# print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: ")
# print(g)
# readline('Program paused. Press enter to continue.')
# 
# # 
# # ================ Part 6: Initializing Pameters ================
# #   In this part of the exercise, you will be starting to implment a two
# #   layer neural network that classifies digits. You will start by
# #   implementing a function to initialize the weights of the neural network
# #   (randInitializeWeights.m)
# 
# print("Initializing Neural Network Parameters ...")
# 
# initial_theta1 <- RandInitializeWeights(input_layer_size, hidden_layer_size)
# initial_theta2 <- RandInitializeWeights(hidden_layer_size, num_labels)
# 
# # Unroll parameters
# initial_nn_params <-c(initial_theta1, initial_theta2)
# 
# # =============== Part 7: Implement Backpropagation ===============
# #     Once your cost matches up with ours, you should proceed to implement the
# #   backpropagation algorithm for the neural network. You should add to the
# #   code you've written in nnCostFunction.m to return the partial
# #   derivatives of the parameters.
# # 
# print("Checking Backpropagation...")
# 
# # Check gradients by running checkNNGradients
# CheckNNGradients()
# readline('Program paused. Press enter to continue.')
# 
# # =============== Part 8: Implement Regularization ===============
# #   Once your backpropagation implementation is correct, you should now
# #   continue to implement the regularization with the cost and gradient.
# # 
# # 
# print("Checking Backpropagation (w/ Regularization) ...")
# # 
# # Check gradients by running checkNNGradients
# lambda <- 3
# CheckNNGradients(lambda)
# 
# # Also output the costFunction debugging values
# debug_J  <- NNCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)$value
# print("Cost at (fixed) debugging parameters (w/ lambda = 3)")
# print(debug_J)
# print("This value should be about 0.576051)")
# readline('Program paused. Press enter to continue.')

# %% =================== Part 8: Training NN ===================
# %  You have now implemented all the code necessary to train a neural 
# %  network. Recall that these advanced optimizers are able to train 
# %  our cost functions efficiently as
# %  long as we provide them with the gradient computations.

print("Training Neural Network...")

# After you have completed the assignment, change the MaxIter to a larger
# value to see how more training helps.
# You should also try different values of lambda
lambda <- 1.0
max_lrate <- 1
relax_factor <- 0.8
num_iter <- 100
# % Create "short hand" for the cost function to be minimized
ObjFunc <- function(p) {
  return (NNCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, TRUE))
}
nn_params <- GradientDescent(ObjFunc, initial_nn_params , max_lrate, num_iter, relax_factor)$params
# Obtain Theta1 and Theta2 back from nn_params
end <- length(nn_params)
theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                 nrow=hidden_layer_size, ncol=(input_layer_size + 1))

theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end],
                 nrow=num_labels, ncol=(hidden_layer_size + 1))
readline('Program paused. Press enter to continue.')

# 
# 
# %% ================= Part 9: Visualize Weights =================
# %  You can now "visualize" what the neural network is learning by 
# %  displaying the hidden units to see what features they are capturing in 
# %  the data.
# 
print("Visualizing Neural Network...")
DisplayData(theta1[, 2:size(theta1,2)]); 
readline('Program paused. Press enter to continue.')

# %% ================= Part 10: Implement Predict =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.
# 
pred <- Predict(theta1, theta2, X)
ps <- mean(pred == y) * 100
print(paste("Train Accuracy", ps, "(%)"))
# Achieve 98.1% with num_iter=1000, relax_factor=0.8, max_lrate=1
readline('Program paused. Press enter to continue.')
