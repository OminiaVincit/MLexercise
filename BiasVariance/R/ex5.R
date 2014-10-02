# Machine Learning Online Class
# Exercise 5 | Regularized Linear Regression and Bias-Variance
# 
#   Instructions
#   ------------
#    
#   This file contains code that helps you get started on the
#   exercise. You will need to complete the following functions:
#   
# %     LinearRegCostFunction.R
# %     LearningCurve.R
# %     ValidationCurve.R
# 
# Initialization
## Initialization
library("R.matlab")
library("pracma")
source("ex5_utils.R")
source("optimizer.R")

# =========== Part 1: Loading and Visualizing Data =============
#     We start the exercise by first loading and visualizing the dataset. 
#   The following code will load the dataset into your environment and plot
#   the data.
#  
# Load Training Data
print("Loading and Visualizing Data ...")

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
pathname <- file.path("../Data", "ex5data1.mat")
data <- readMat(pathname)
X <- data$X
y <- data$y
Xval <- data$Xval
yval <- data$yval
Xtest <- data$Xtest
ytest <- data$ytest
m <- size(X, 1)

# Plot training data
plot(x=X[,1], y=y[,1], type="p", col="red", pch=16, xlab="Change in water level(x)", 
     ylab="Water flowing out of the dam (y)",main="")

readline('Program paused. Press enter to continue.')

# =========== Part 2: Regularized Linear Regression Cost & Gradient =============
#   You should now implement the cost function for regularized linear 
#   regression. 

theta <- c(1, 1)
J <- LinearRegCostFunction(cbind(ones(m,1),X), y, theta, 1, TRUE);
print(paste("Cost at theta = (1,1) is ", J$value))
print("This value should be about 303.993192")
print("Gradient at theta = (1,1) is ")
print(J$grad)
print("This value should be about [-15.303016; 598.250744]")
readline('Program paused. Press enter to continue.')

# %% =========== Part 4: Train Linear Regression =============
#   % Once you have implemented the cost and gradient correctly, the
# %  trainLinearReg function will use your cost function to train 
# %  regularized linear regression.
# % 
# %  Write Up Note: The data is non-linear, so this will not give a great 
# %                 fit.
# %
# 
# Train linear regression with lambda = 0
lambda <- 0
theta <- TrainLinearReg(cbind(ones(m, 1), X), y, lambda)

# Plot fit over the data
lines(x=X[,1], y=cbind(ones(m, 1), X)%*%theta, col="blue", pch=16)
readline('Program paused. Press enter to continue.')

# %% =========== Part 5: Learning Curve for Linear Regression =============
#   %  Next, you should implement the learningCurve function. 
# %
# %  Write Up Note: Since the model is underfitting the data, we expect to
# %                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
# %
# 
lambda <- 0
C <- LearningCurve(cbind(ones(m, 1), X), y, cbind(ones(size(Xval, 1), 1), Xval), yval, lambda)
plot(x=1:m, y=C$error_train, type="l", col="purple", 
     xlab="Number of training examples", ylab="Error",
     xlim=c(0,13), ylim=c(0,150)) 
lines(x=1:m, y=C$error_val, type="l", col="green")
legend("topright", inset=.02, c("Train","Validation"), cex=.8, col=c("purple","green"), pch=c("-","-") )
readline('Program paused. Press enter to continue.')

# 
# %% =========== Part 6: Feature Mapping for Polynomial Regression =============
#   %  One solution to this is to use polynomial regression. You should now
# %  complete polyFeatures to map each example into its powers

p <- 5
# % Map X onto Polynomial Features and Normalize
X_poly <- PolyFeatures(X, p)
NM <- FeatureNormalize(X_poly)  # Normalize
X_poly <- NM$norm
sigma <- NM$sigma
mu <- NM$mu

X_poly <- cbind(ones(m, 1), X_poly) #Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test <- PolyFeatures(Xtest, p)
X_poly_test <- BFXFunc("minus", X_poly_test, mu)
X_poly_test <- BFXFunc("divide", X_poly_test, sigma)
X_poly_test <- cbind(ones(size(X_poly_test,1),1), X_poly_test) #Add ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val <- PolyFeatures(Xval, p)
X_poly_val <- BFXFunc("minus", X_poly_val, mu)
X_poly_val <- BFXFunc("divide", X_poly_val, sigma)
X_poly_val <- cbind(ones(size(X_poly_val,1),1), X_poly_val) #Add ones

print("Normalized Training Example 1")
print(X_poly[1,])
# 
# %% =========== Part 7: Learning Curve for Polynomial Regression =============
#   %  Now, you will get to experiment with polynomial regression with multiple
# %  values of lambda. The code below runs polynomial regression with 
# %  lambda = 0. You should try running the code with different values of
# %  lambda to see how the fit and learning curve change.

lambda <- 3
theta <- TrainLinearReg(X_poly, y, lambda)

# Plot fit over the data
plot(x=X[,1], y=y[,1], type="p", col="red", pch=16, xlab="Change in water level(x)", 
     ylab="Water flowing out of the dam (y)",main=paste("Polynomial Regression Fit, lambda=",lambda))
PlotFit(min(X[,1]), max(X[,1]), mu, sigma, theta, p)
readline('Program paused. Press enter to continue.')

C <- LearningCurve(X_poly, y, X_poly_val, yval, lambda)
plot(x=1:m, y=C$error_train, type="l", col="purple", xlim=c(0,13), ylim=c(0,100),
     xlab="Number of training examples", ylab="Error", 
     main=paste("Polynomial Regression Learning Curve, lambda=",lambda)) 
lines(x=1:m, y=C$error_val, type="l", col="green")
legend("topright", inset=.02, c("Train","Validation"), cex=.8, col=c("purple","green"), pch=c("-","-") )
readline('Program paused. Press enter to continue.')

# 
# %% =========== Part 8: Validation for Selecting Lambda =============
#   %  You will now implement validationCurve to test various values of 
# %  lambda on a validation set. You will then use this to select the
# %  "best" lambda value.
# %
VC <- ValidationCurve(X_poly, y, X_poly_val, yval)
plot(x=VC$lambda_vec, y=VC$error_train, type="l", col="purple", xlim=c(0,13), ylim=c(0,100),
     xlab="Lambda", ylab="Error") 
lines(x=VC$lambda_vec, y=VC$error_val, type="l", col="green")
legend("topright", inset=.02, c("Train","Validation"), cex=.8, col=c("purple","green"), pch=c("-","-") )
readline('Program paused. Press enter to continue.')
