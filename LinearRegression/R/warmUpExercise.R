MakeMatrix <- function(n) {
  # WARMUPEXERCISE Example function in R
  #   A = WARMUPEXERCISE(n) is an example function that returns the nxn identity matrix
  # ============= YOUR CODE HERE ==============
  # Instructions: Return the nxn identity matrix 
  A <- diag(n)
  return (A)
  # ===========================================
}

PlotData <- function(xData, yData){
  # PLOTDATA Plots the data points x and y into a new figure 
  #   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
  #   population and profit.
  
  # ====================== YOUR CODE HERE ======================
  
  plot(x=xData, y=yData, type="p", xlab="Population of City in 10,000s", ylab="Profit in $10,000s", 
       main="Scatter plot of training data", col="red")
  
  # ============================================================
}
