## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
  # 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.

## Initialization
source("ex2_utils.R")
library("trust")

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print("Loading data")
filename <- paste("../Data/ex2data2.txt")
rawData <- read.csv(filename, header=F)
X <- rawData[,1:2]
y <- rawData[,3]
PlotData(X,y)
# Put some Labels and Legend
title(xlab="Microchip Test 1", ylab="Microchip Test 2", main="Scatter plot of  trainning data")
legend("topright", inset=.02, c("y=0","y=1"),cex=.8, col=c("red","blue"), pch=c(1,2))
       
# =========== Part 1: Regularized Logistic Regression ============
#   In this part, you are given a dataset with data points that are not
#    linearly separable. However, you would still like to use logistic 
#    regression to classify the data points. 
# 
# To do so, you introduce more features to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial regression).
# 
# Add Polynomial Features
# 
# Note that mapFeature also adds a column of ones for us, 
# so the intercept term is handled
X <- MapFeature(X[,1], X[,2])

# Initialize fitting parameters
initial_theta <- rep(0, length(X[1,]))

# Set regularization parameter lambda to 1
lambda <- 1
# 
# Compute and display initial cost and gradient for regularized logistic regression
rs <- CostFunction(X, y, initial_theta, lambda)
cost <- rs$value
grad <- rs$gradient
hess <- rs$hessian
print("Cost at initial theta (zeros)")
print(cost)
print("Gradient at initial theta (zeros)")
print(grad)
print("Hessian at initial theta (zeros)")
print(hess)
readline('Program paused. Press enter to continue.')
# 
# % ============= Part 2: Regularization and Accuracies =============
#   %  Optional Exercise:
#   %  In this part, you will get to try different values of lambda and 
# %  see how regularization affects the decision coundart
# %
# %  Try the following values of lambda (0, 1, 10, 100).
# %
# %  How does the decision boundary change when you vary lambda? How does
# %  the training set accuracy vary?
# %
# 
# % Set regularization parameter lambda to 1 (you should vary this)
lambdas <- c(0, 1, 10, 100)
num_lambda <- length(lambdas)
thetas <- diag(0, length(X[1,]), num_lambda)
# Plot 4 graphs in one window
old.par <- par(mfrow=c(2,2))

for (i in 1:num_lambda){
  initial_theta <- rep(0, length(X[1,]))
  ObjFunc <- function(th) {
    return (CostFunction(X, y, th, lambdas[i]))
  }
  whoop <- trust(ObjFunc, initial_theta, 1, 100, blather=TRUE)
  thetas[,i] <- whoop$argument
  PlotDecisionBoundary(thetas[,i], X, y)
  title(xlab="Microchip Test 1", ylab="Microchip Test 2", main=paste("lambda = ", lambdas[i]) )
  #legend("topright", inset=.02, c("y=0","y=1"), cex=.8, col=c("red","blue"), pch=c(1,2) )
  #legend("topleft", inset=.02, c("Decision boundary"), cex=.8, col="green", pch="-" )
  
  # Compute accuracy on our training set
  p <- Predict(thetas[,i], X)
  ps <- mean(p == y) * 100
  print(paste("lambda =", lambdas[i], " with train Accuracy = ", ps, "%"))
}
par(old.par)
readline('Program paused. Press enter to continue.')