## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. 
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## Initialization
source("warmUpExercise.R")
source("utils.R")

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.R
print('Running MakeMatrix ...')
print('5x5 Identity Matrix: ')

A = MakeMatrix(5)
readline('Program paused. Press enter to continue.\n');

## ==================== Part 2: Plotting ====================
# Draw multi plot in one window

# Load data
filename <- paste("../Data/ex1data1.txt")
rawdata <- read.csv(filename,header=F)
xData <- rawdata[,1]
yData <- rawdata[,2]
m <- length(yData)

print('Running PlotData ...')
old.par <- par(mfrow=c(2,2))
PlotData(xData,yData)
readline('Program paused. Press enter to continue.\n');

## ==================== Part 3: Gradient descent ====================
print('Running Gradient Descent ...')
X <- matrix(c( rep(1,m), xData), nrow=m,ncol=2) # Add a column of ones to x

# Some gradient descent setting
iterations <- 2000
alpha <- 0.01
theta <- c(0,0)

# Compute and display initial cost
J <- ComputeCost(X, yData, theta)

# Run the gradient descent
result <- GradientDescent(X, yData, theta, alpha, iterations)
theta <- result$theta

# Print theta
print("Theta found by gradient descent: ")
print(theta)

# print("J_history for debug")
# J_history <- result$costs
# print(J_history)

# Plot the linear fit
# Hold on and keep the previous plot visible
lines(xData, X%*%theta,type="l",col="blue")
# Hold off, don't overlay any more plots on this figure

# Predict values for population size of 35,000 and 70,000
predict1 <- t(theta) %*% c(1,3.5)
print("For population = 35,000, we predict a profit of : ")
print(predict1)

predict2 <- t(theta) %*% c(1,7)
print("For population = 70,000, we predict a profit of : ")
print(predict2)

readline('Program paused. Press enter to continue.\n')

## ======= Part 4: Visualizing cost function J(theta0, theta1) ==========
# Surface
gridlen <- 100
theta0_vals <- seq(-10,10,20/gridlen)  # Set grid for x dimension
theta1_vals <- seq(-1, 4, 5/gridlen) # Set grid for y dimension
len0 <- length(theta0_vals)
len1 <- length(theta1_vals)
J_vals <- diag(0, len0, len1)
for (i in 1:len0){
  for (j in 1:len1){
    J_vals[i,j] <- ComputeCost(X, yData, c(theta0_vals[i], theta1_vals[j]))
  }
}
contour(theta0_vals, theta1_vals, log(J_vals), col = "blue" , lwd=2, main="Contour plot", xlab="theta0", ylab="theta1")
points(theta[1], theta[2], type="p", col="red", pch=21 )

persp(theta0_vals, theta1_vals, J_vals, theta=40, phi=45, col=rainbow(40), shade=.6, ticktype="detailed",
       xlab="theta0", ylab="theta1", zlab="Cost", main="Surface cost function")

par(old.par)
readline('Program paused. Press enter to continue.\n')
