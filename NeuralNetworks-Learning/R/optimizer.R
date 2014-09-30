# We can deal with many methods for optimizer (LBFGS, ...)
GradientDescent <- function(fobj, params, max_lrate, max_iter, relax_factor, epsilon=1e-8, min_lrate=1e-4){
  cost <- 0
  curr_grad <- rep(0, length(params))
  prev_grad <- rep(0, length(params))
  lrate <- max_lrate
  for(i in 1:max_iter){
    rs <- fobj(params)
    cost <- rs$value
    prev_grad <- curr_grad
    curr_grad <- rs$grad
    magnitude <- sqrt(sum(curr_grad*curr_grad))
    if (magnitude < epsilon || lrate < min_lrate) break
    direction <- sum(prev_grad*curr_grad)
    if (direction < 0){
      # change length of step
      lrate <- lrate*relax_factor
    }
    params <- params - (lrate/magnitude)*curr_grad
    print(paste("Iteration ",i,", lrate = ", lrate, ", cost = ",cost))
  }
  return(list(value=cost,params=params))
}