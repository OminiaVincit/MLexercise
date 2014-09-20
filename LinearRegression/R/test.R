# # Read data first
# 
# # Filename list
# name <- c(7,2,14,4,18,8,12,20,11,10,19)
# 
# # Define standard data set
# # Interval 1 hour
# # 9 days: 23/7 - 31/7
# # Number of data points = 9 x 24 = 216 points
# standNum <- 216
# 
# metric <- NULL
# metric$Id  <- 1:standNum
# metric$Val <- 1:standNum
# metric$Num <- 1:standNum
# 
# result <- NULL
# result$Id <- 1:standNum
# result$Num <- 0*(1:standNum)
# 
# start <- "2014-07-23 00:00:00"
# st_x = as.numeric(as.POSIXct(start))
# 
# for ( j in 1:11 ) {
#   filename <- paste("./csvs_analysis/WP",name[j],"Motion.csv",sep="")
#   rawData <- read.csv(filename)
#   xData <- rawData[,1]
#   yData <- rawData[,2]
#   realNum <-length(xData)
#   
#   # Initialize
#   for ( i in 1:standNum ){
#     metric$Val[i] = 0;
#     metric$Num[i] = 0;
#   }
#   
#   # Get into interval
#   for ( i in 1:realNum ) {
#     # get the time
#     # convert to inverval
#     y = as.numeric(as.POSIXct(xData[i]))
#     tmp <- ( y - st_x) / 3600
#     id <- floor( tmp ) + 1
#     metric$Val[id] <- (metric$Val[id] + yData[i])
#     metric$Num[id] <- (metric$Num[id] + 1)
#   }
#   
#   for ( i in 1:standNum ){
#     if ( metric$Num[i] > 0 ) {
#       ds <- metric$Val[i]
#       dn <- metric$Num[i]
#       metric$Val[i] <- ds/dn
#       if (metric$Val[i] > 0.5) result$Num[i] = result$Num[i] + 1
#     }
#   }  
# }

#
#plot(x=xData, y=yData, type="l", xlab="time", ylab="value", main="")

# Test for contour
library (rsm)
heli.rsm = rsm (ave ~ block + SO(x1, x2, x3, x4), data = heli)

# Plain contour plots
par (mfrow = c (2,3))
contour (heli.rsm, ~x1+x2+x3+x4, at=canonical(heli.rsm)$xs)

# Same but with image overlay, slices at origin and block 2,
# and no slice labeling
contour (heli.rsm, ~x1+x2+x3+x4, at=list(block="2"), atpos=0, image=TRUE)

# Default perspective views
persp (heli.rsm, ~x1+x2+x3+x4, at=canonical(heli.rsm)$xs)

# Same plots, souped-up with facet coloring and axis labeling
persp (heli.rsm, ~x1+x2+x3+x4, contours="col", col=rainbow(40), at=canonical(heli.rsm)$xs,
       xlabs = c("Wing area", "Wing length", "Body width", "Body length"), zlab = "Flight time")

## Not run: 
### Hints for creating graphics files for use in publications...

# Save perspective plots in one PDF file (will be six pages long)
# pdf(file = "heli-plots.pdf")
# persp (heli.rsm, ~x1+x2+x3+x4, at=canonical(heli.rsm)$xs)
# dev.off()

# Save perspective plots in six separate PNG files
# png.hook = list()
# png.hook$pre.plot = function(lab) 
#   png(file = paste(lab[3], lab[4], ".png", sep = ""))
# png.hook$post.plot = function(lab)
#   dev.off()
# persp (heli.rsm, ~x1+x2+x3+x4, at=canonical(heli.rsm)$xs, hook = png.hook)

## End(Not run) 