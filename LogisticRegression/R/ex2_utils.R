PlotData <- function(X, y){
  #   PLOTDATA(x,y) plots the data X with label y
  pos <- which(y==1)
  neg <- which(y==0)
  plot(x=X[pos,1], y=X[pos,2], type="p", col="red", pch=1,xlab="",ylab="",main="")
  points(x=X[neg,1], y=X[neg,2], type='p', col="blue", pch=2)
}

PlotDecisionBoundary <- function(theta, X, y){
  # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
  # the decision boundary defined by theta
  #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
  #   positive examples and o for the negative examples. X is assumed to be 
  #   a either 
  #   1) Mx3 matrix, where the first column is an all-ones column for the 
  #      intercept.
  #   2) MxN, N>3 matrix, where the first column is all-ones
  
  pos <- which(y==1)
  neg <- which(y==0)
  plot(x=X[pos,2], y=X[pos,3], xlim=c(-1,1.5), ylim=c(-1,1.5), type="p", col="red", pch=1,xlab="",ylab="",main="")
  points(x=X[neg,2], y=X[neg,3], type='p', col="blue", pch=2)
  
  if (length(X[1,]) <= 3) {
    # Ony need 2 points to define a line, so choose two endpoints
    plot_x = c( min(X[,2])-2, max(X[,2])+2 )
    # Calculate the decision boundary line
    plot_y = (-1.0/theta[3])*(theta[2]*plot_x + theta[1])
    lines(plot_x, plot_y, col="green")
    legend("topright", inset=.02, c("Admitted","Not-admitted"), cex=.8, col=c("red","blue"), pch=c(1,2) )
    legend("topleft", inset=.02, c("Decision boundary"), cex=.8, col="green", pch="-" )
  }
  else {
    # The grid range
    u <- seq(-1, 1.5, 0.05)
    v <- seq(-1, 1.5, 0.05)
    z <- diag(0, length(u), length(v))
    # Evaluate z = theta*x over the grid
    for (i in 1:length(u)){
      for (j in 1:length(v)){
        z[i,j] <- MapFeature(u[i], v[j]) %*% theta
      }
    }
    # Transpose z before calling contour
    z <- t(z)
    # Plot z = 0
    contour(u, v, z, zlim=c(-1e-10,1e-10),col = "green", lwd=2, add=TRUE)
  }
}

Sigmod <- function(theta, x) {
  z <- t(theta)%*%x
  rs <-  1/(1+exp(-z))
  return (rs[1,1])
}

Predict <- function(theta, X) {
  # PREDICT Predict whether the label is 0 or 1 using learned logistic
  # regression parameters theta
  # p = PREDICT(theta,X) computes the prediction for X using a 
  # threshold at 0.5 (i.e., if sigmod(T(theta)*x) >= 0.5, predict 1)
  
  m <- length(X[,1]) # Number of trainning examples
  p <- rep(0, m)
  for (i in 1:m){
    if (Sigmod(theta,X[i,]) >= 0.5) {
      p[i] <- 1
    }
  }
  return (p)
}

CostFunction <- function(X, y, theta, lambda=0) {
  # COSTFUNCTIONREG Compute costa and gradient, hessian for logistic regression
  # and with regularization if lambda != 0
  #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  #   theta as the parameter for regularized logistic regression and the
  #   gradient, the hessian of the cost w.r.t. to the parameters. 
  
  # Initialize some useful values
  m <- length(y) # number of training examples
  n <- length(X[1,])
  sigs <- c(1:m)*0
  for (i in 1:m) {
    sigs[i] <- Sigmod(theta,X[i,])
  }
  cost <- sum(-y*log(sigs) - (1-y)*log(1-sigs))/m
  grad <- rep(0,n)
  B <- diag(0,n)
  ts_sum <- sum( sigs*(1-sigs) )/m
  for (j in 1:n) {
    grad[j] <- sum( (sigs-y)*X[,j] )/m 
    for (k in 1:n){
      B[j,k] <- sum( sigs*(1-sigs)*X[,j]*X[,k] )/m
    }
  }
  if (lambda != 0){
    # Regularization case
    cost <- cost + lambda*sum(theta^2)/(2*m)
    grad <- grad + lambda*c(0,theta[2:n])/m
    B <- B + diag(c(0,rep(lambda/m,n-1)),n)
  }
  return(list(value = cost, gradient = grad, hessian = B))
}

MapFeature <- function(X1,X2) {
  # MAPFEATURE Feature mapping function to polynomial features
  # 
  #   MAPFEATURE(X1,X2) maps the two input features
  #   to quadratic features used in the regularization exercise
  #
  #   Returns a new feature array with more features, comprising of
  #   X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc...
  #
  #   Inputs X1, X2 must be the same size
  
  degree <- 6
  m <- length(X1)
  out <- matrix(rep(1,m), nrow=m, ncol=1)
  for(i in 1:degree){
    for(j in 0:i){ 
      out <- cbind(out,(X1^(i-j))*(X2^j))
    }
  }
  return(out)
}