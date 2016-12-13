library(data.table)
library(ggplot2)

#======================================================================================================
# Helper method for generating a random ellipse

rEllipse <- function(n=1, width=1, height=1, center=c(x=0, y=0)){
  # Generate random points inside an ellipse
  
  rho <- runif(n=n)
  phi <- runif(n=n, min=0, max=2*pi)
  
  x <- (sqrt(rho) * cos(phi)) * width / 2 + center[1]
  y <- (sqrt(rho) * sin(phi)) * height / 2 + center[2]
  
  result <- data.table(x=x, y=y)
  return(result)
}

#======================================================================================================
# Make data

set.seed(1)

bob <- rbind(
  rEllipse(5, width=2, height=2, center=c(0, 0)),
  rEllipse(210, width=2, height=2, center=c(0, 0))[x^2 + y^2 > .9^2]
)

sue <- rbind(
  rEllipse(5, width=2, height=2, center=c(0, 0)),
  rEllipse(210, width=2, height=2, center=c(0, 0))[between(x^2 + y^2, .7^2, .9^2)]
)[sample(.N, 50)]

mark <- rbind(
  rEllipse(10, width=2, height=2, center=c(0, 0)),
  rEllipse(1000, width=2, height=2, center=c(0, 0))[between(x, -.275, .275) & between(y, -.6, .6)]
)[sample(.N, 50)]

kate <- rbind(
  rEllipse(10, width=2, height=2, center=c(0, 0)),
  rEllipse(3, width=.2, height=.1, center=c(.3, -.6)),
  rEllipse(4, width=.2, height=.1, center=c(.3, -.3)),
  rEllipse(3, width=.2, height=.1, center=c(.3, 0)),
  rEllipse(4, width=.2, height=.1, center=c(.3, .3)),
  rEllipse(3, width=.2, height=.1, center=c(.3, .6)),
  rEllipse(4, width=.2, height=.1, center=c(-.3, -.6)),
  rEllipse(3, width=.2, height=.1, center=c(-.3, -.3)),
  rEllipse(4, width=.2, height=.1, center=c(-.3, 0)),
  rEllipse(3, width=.2, height=.1, center=c(-.3, .3)),
  rEllipse(4, width=.2, height=.1, center=c(-.3, .6))
)

#--------------------------------------------------
# Combine competitor throws

dt <- rbind(
  bob[, list(x, y, Competitor="Bob")],
  sue[, list(x, y, Competitor="Sue")],
  mark[, list(x, y, Competitor="Mark")],
  kate[, list(x, y, Competitor="Kate")]
)

# Check distribution
dt[, .N, keyby=Competitor]

# Adjust column names
setnames(dt, c("x", "y"), c("XCoord", "YCoord"))

# Ranom Shuffle
dt <- dt[sample(.N, .N)]

# Make Competitor a factor
dt[, Competitor := factor(Competitor)]

# Add ID column
dt[, ID := .I]

# Fix column order
setcolorder(dt, c("ID", "XCoord", "YCoord", "Competitor"))

# Plot
ggplot(dt, aes(x=XCoord, y=YCoord, color=Competitor))+geom_point()

#======================================================================================================
# Partition data

#--------------------------------------------------
# Split into train and test

train <- dt[sample(.N, round(nrow(dt) * .8))]
test <- dt[!train, on="ID"]

#======================================================================================================
# Write Files to disc

# fwrite(train, "/Users/Ben/Businesses/GormAnalysis/Internal Projects/MLPB/Problems/Classify Dart Throwers/_Data/train.csv")
# fwrite(test, "/Users/Ben/Businesses/GormAnalysis/Internal Projects/MLPB/Problems/Classify Dart Throwers/_Data/test.csv")
