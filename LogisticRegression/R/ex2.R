## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
  # 
#  This file contains code that helps you get started on the logistic
#  regression exercise. 

## Initialization
source("ex2_utils.R")
library("trust")

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print("Loading data")
filename <- paste("../Data/ex2data1.txt")
rawData <- read.csv(filename, header=F)
X <- rawData[,1:2]
y <- rawData[,3]

## ==================== Part 1: Plotting ====================
  #  We start the exercise by first plotting the data to understand the the problem we are working with.

print("Plotting data with + indicating (y=1) examples and o indicating (y=0) examples")
PlotData(X, y)
title(xlab="Exam 1 score", ylab="Exam 2 score", main="Scatter plot of  trainning data")
readline('Program paused. Press enter to continue.')

# ## ============ Part 2: Compute Cost and Gradient ============
#   #  In this part of the exercise, you will implement the cost and gradient
# #  for logistic regression. You neeed to complete the code in 

# Setup the data matrix appropriately, and add ones for the intercept term
# Add intercept term (column 1) to X
m <- length(X[,1])
n <- length(X[1,])
size <- m*(n+1)
mat <- matrix(rep(1,size), nrow=m, ncol=n+1)
for (i in 1:n) {
  mat[,i+1] <- X[,i]
}
X <- mat

# Initialize fitting parameters
initial_theta = rep(0, n+1)

# Compute and display initial cost and gradient
rs <- CostFunction(X, y, initial_theta)
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
# 
# ## ============= Part 3: Optimizing using trust  =============
#   In this exercise, you will use a built-in function (trust) to find the
#   optimal parameters theta.
# 
ObjFunc <- function(th) {
  return (CostFunction(X,y,th))
}
whoop <- trust(ObjFunc, initial_theta, 1, 100, blather=TRUE)

# Print theta to screen
print('Cost at theta found by trust')
theta <- whoop$argument
print(theta)
# 
# Plot Boundary
PlotDecisionBoundary(theta, X, y)
title(xlab="Exam 1 score", ylab="Exam 2 score", main="Scatter plot of  trainning data")
readline('Program paused. Press enter to continue.')

# 
# %% ============== Part 4: Predict and Accuracies ==============
#   %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and 
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of 
# %  our model.

# %  Predict probability for a student with score 45 on exam 1 
# %  and score 85 on exam 2 
 
prob <- Sigmod(theta, c(1, 45, 85))
print('For a student with scores 45 and 85, we predict an admission probability of')
print(prob)

# 
# Compute accuracy on our training set
p <- Predict(theta, X)
ps <- mean(p == y) * 100
print('Train Accuracy (%)')
print(ps)
readline('Program paused. Press enter to continue.')
