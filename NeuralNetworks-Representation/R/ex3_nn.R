## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
  
## Initialization
library("R.matlab")
library("pracma")
source("ex3_utils.R")

## Setup the parameters you will use for this exercise
input_layer_size  <- 400 # 20x20 Input Images of Digits
hidden_layer_size <- 25  # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
# (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
  #  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

print('Loading and Visualizing Data ...')
pathname <- file.path("../Data", "ex3data1.mat")
data <- readMat(pathname) # training data stored in arrays X, y
X <- data$X
y <- data$y
m <- length(X[,1])
n <- length(X[1,])

# Randomly select 100 data points to display
rand_indices <- randperm(m,100)
sel <- X[rand_indices, ]
DisplayData(sel)
readline('Program paused. Press enter to continue.')

## ================ Part 2: Loading Pameters ================
  # In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...')

# Load the weights (trained) into variables Theta1 and Theta2
pathname <- file.path("../Data", "ex3weights.mat")
data <- readMat(pathname)
theta1 <- data$Theta1 # size 25 x 401
theta2 <- data$Theta2 # size 10 x 26

## ================= Part 3: Implement Predict =================
  #  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred <- Predict(theta1, theta2, X)
ps <- mean(pred == y) * 100
print(paste("Train Accuracy", ps, "(%)"))
# "Train Accuracy 97.52 (%)"
readline('Program paused. Press enter to continue.')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp <- randperm(m,m)
 
for (i in 1:m){
  # Display 
  print(paste("Displaying Example Image",i))
  tmp <- t(X[rp[i], ])
  DisplayData(tmp)
  pred <- Predict(theta1, theta2, tmp)
  print(paste("Neural Network Prediction: ", pred, ",digit ", mod(pred, 10)))
  readline('Program paused. Press enter to continue.')
}

