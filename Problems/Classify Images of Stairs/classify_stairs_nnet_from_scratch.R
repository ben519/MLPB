# Datasets and graphs for the article Neural Networks - A Worked Example
# https://gormanalysis.com/neural-networks-a-worked-example

#======================================================================================================
# settings

options(scipen = 20)

#--------------------------------------------------
# packages

library(data.table)
library(ggplot2)
library(stringr)

#======================================================================================================
# Helper methods

sigmoid <- function(x){
  # sigmoid function
  
  1 / (1 + exp(-x))
}

softmax <- function(m, offset_trick = TRUE){
  # softmax
  
  if(offset_trick){
    rowmaxs <- apply(m, 1, max)
    result <- exp(m - rowmaxs)/rowSums(exp(m - rowmaxs))
  } else{
    result <- exp(m)/rowSums(exp(m))
  }
  
  return(result)
}

#======================================================================================================
# Datasets

# Read train & test data
train <- fread("~/Projects/R/MLPB/Problems/Classify Images of Stairs/_Data/train.csv")
test <- fread("~/Projects/R/MLPB/Problems/Classify Images of Stairs/_Data/test.csv")

# Insert Label for plots
train[, Label := ifelse(IsStairs, "Stairs", "Not Stairs")]
train[, Label := factor(Label, levels=c("Not Stairs", "Stairs"))]
test[, Label := ifelse(IsStairs, "Stairs", "Not Stairs")]
test[, Label := factor(Label, levels=c("Not Stairs", "Stairs"))]

# Reshape from wide to tall
trainTall <- melt(
  train, 
  id.vars = c("ImageId", "IsStairs", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)

testTall <- melt(
  test, 
  id.vars = c("ImageId", "IsStairs", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)

trainTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
testTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]

#======================================================================================================
# Figures

#--------------------------------------------------
# fig1: stairs vs not stairs

fig1data <- trainTall[ImageId %in% train$ImageId[1:12]]
fig1 <- ggplot(fig1data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+
  facet_wrap(~ImageId, nrow = 2, labeller = labeller(ImageId = setNames(as.character(fig1data$Label), fig1data$ImageId)))

#--------------------------------------------------
# fig2: image with pixels labeled x1, x2, x3, x4

image1 <- trainTall[ImageId == 1]
image1[Row == 1 & Col == 1, PixelLabel := paste0("x1 = ", Intensity)]
image1[Row == 1 & Col == 2, PixelLabel := paste0("x2 = ", Intensity)]
image1[Row == 2 & Col == 1, PixelLabel := paste0("x3 = ", Intensity)]
image1[Row == 2 & Col == 2, PixelLabel := paste0("x4 = ", Intensity)]

fig2lables <- image1
fig2lables[, `:=`(x = Col+0.45, y = 1.55 - Row)]

fig2 <- ggplot(image1, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  geom_text(data = fig2lables, aes(x = x, y = y, label = PixelLabel), color = "goldenrod3", size = 5, hjust = 1, vjust = 0)+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+
  labs(title = image1$Label[1])+
  theme(plot.title = element_text(hjust = 0.5, vjust = -50))

#======================================================================================================
# NNet from scratch

# Build a nnet with and input layer, hidden layer, and output layer
# input layer: 1st node = 1 (for bias), nodes 2-4 correspond to features R1C1, R1C2, R2C1, R2C2
# hidden layer: 3 nodes. 1st node = 1 (for bias), nodes 2-3 correspond to incoming signals
# output layer: 2 nodes
# Optimize categorical cross entropy error
# Apply sigmoid activation to the hidden layer, softmax to the output layer

#--------------------------------------------------
# Set Y, X1, stepsize

Y <- cbind(Stairs = 1 * train$IsStairs, NotStairs = 1 * !train$IsStairs)
X1 <- as.matrix(train[, list(R1C1, R1C2, R2C1, R2C2)])

# Prepend X1 with column of 1s for bias terms
X1 <- cbind(b = 1, X1)  # Include column of all 1s

# Set setpsize
stepsize = 0.1

#--------------------------------------------------
# Weight initialization

set.seed(1)
W1 <- matrix(runif(min = -0.01, max = 0.01, n = 5*2), nrow=5)
W2 <- matrix(runif(min = -0.01, max = 0.01, n = 3*2), nrow=3)

#--------------------------------------------------
# Forward Pass (first iteration)

Z1 <- X1 %*% W1
X2 <- cbind(b = 1, sigmoid(Z1))
Z2 <- X2 %*% W2
Yhat <- softmax(Z2)

# Calculate Categorical Cross Entropy Error
loss <- mean(-rowSums(Y * log(Yhat)))

# Measure loss and classification accuracy
predicted_class <- apply(Yhat, MARGIN = 1, FUN = which.max)
true_class <- apply(Y, MARGIN = 1, FUN = which.max)
accuracy <- mean(predicted_class == true_class)
print(paste("Loss:", loss, "| accuracy:", accuracy))

#--------------------------------------------------
# Backprop (first iteration)

# Backprop
delta1 <- Yhat - Y  # Partial CE/Partial Z2
delta2 <- delta1 %*% t(W2)  # Partial CE/Partial X2
delta3 <- delta2[, -1] * (X2[, -1] * (1 - X2[, -1]))  # Partial CE/Partial Z1

# Gradients
gradW2 <- t(X2) %*% delta1 / nrow(X1)
gradW1 <- t(X1) %*% delta3 / nrow(X1)

# Weight updates
W1 <- W1 - gradW1 * stepsize
W2 <- W2 - gradW2 * stepsize

# Measure loss and classification accuracy
predicted_class <- apply(Yhat, MARGIN = 1, FUN = which.max)
true_class <- apply(Y, MARGIN = 1, FUN = which.max)
accuracy <- mean(predicted_class == true_class)
print(paste("Loss:", loss, "| accuracy on training data:", accuracy))

#--------------------------------------------------
# Repeat

epochs <- 1500
for(epoch in 2:epochs){
  
  # Forward Pass
  Z1 <- X1 %*% W1
  X2 <- cbind(b = 1, sigmoid(Z1))
  Z2 <- X2 %*% W2
  Yhat <- softmax(Z2)
  
  # Calculate Categorical Cross Entropy Error
  loss <- mean(-rowSums(Y * log(Yhat)))
  
  # Backprop
  delta1 <- Yhat - Y  # Partial CE/Partial Z2
  delta2 <- delta1 %*% t(W2[-1, ])  # Partial CE/Partial X2[, -1]
  delta3 <- delta2 * (X2[, -1] * (1 - X2[, -1]))  # Partial CE/Partial Z1
  
  # Gradients
  gradW2 <- t(X2) %*% delta1 / 400
  gradW1 <- t(X1) %*% delta3 / 400
  
  # Weight updates
  W1 <- W1 - gradW1 * stepsize
  W2 <- W2 - gradW2 * stepsize
  
  # Print the loss and accuracy
  if((epoch - 2) %% 100 == 0){
    predicted_class <- apply(Yhat, MARGIN = 1, FUN = which.max)
    true_class <- apply(Y, MARGIN = 1, FUN = which.max)
    accuracy <- mean(predicted_class == true_class)
    
    print(paste("Epoch:", epoch, "Loss:", loss, "| accuracy on training data:", accuracy))
  }
}

#--------------------------------------------------
# Make predictions on the test data

Y_test <- cbind(Stairs = 1 * test$IsStairs, NotStairs = 1 * !test$IsStairs)
X1_test <- as.matrix(test[, list(R1C1, R1C2, R2C1, R2C2)])

# Prepend X1 with column of 1s for bias terms
X1_test <- cbind(b = 1, X1_test)  # Include column of all 1s

# Forward pass
Z1_test <- X1_test %*% W1
X2_test <- cbind(b = 1, sigmoid(Z1_test))
Z2_test <- X2_test %*% W2
Yhat_test <- softmax(Z2_test)

# Check results
predicted_class <- apply(Yhat_test, MARGIN = 1, FUN = which.max)
true_class <- apply(Y_test, MARGIN = 1, FUN = which.max)
accuracy <- mean(predicted_class == true_class)
print(paste("accuracy on test data:", accuracy))  # 0.87

##################################################################################################################################
# Check results using keras

library(keras)

Y <- cbind(Stairs = 1 * train$IsStairs, NotStairs = 1 * !train$IsStairs)
X1 <- as.matrix(train[, list(R1C1, R1C2, R2C1, R2C2)])

set.seed(1)
W1 <- matrix(runif(min = -0.01, max = 0.01, n = 5*2), nrow=5)
W2 <- matrix(runif(min = -0.01, max = 0.01, n = 3*2), nrow=3)

model <- keras_model_sequential()
model <- layer_dense(
  object = model, 
  input_shape = 4L, 
  use_bias = TRUE, 
  units = 2L, 
  activation = 'sigmoid', 
  weights = list(W1[2:5,], array(W1[1,]))
)
model <- layer_dense(
  object = model, 
  input_shape = 2L, 
  use_bias = TRUE,
  units = 2L,
  activation = 'softmax',
  weights = list(W2[2:3,], as.array(W2[1,]))
)
model <- compile(
  object = model, 
  loss = 'categorical_crossentropy', 
  optimizer = optimizer_sgd(lr=0.1), 
  metrics = c('accuracy')
)

# Check the weights
get_weights(model)

# [[1]]
# [,1]         [,2]
# [1,] -0.002557522  0.008893506
# [2,]  0.001457067  0.003215956
# [3,]  0.008164156  0.002582281
# [4,] -0.005966362 -0.008764274
# 
# [[2]]
# [1] -0.004689827  0.007967793
# 
# [[3]]
# [,1]           [,2]
# [1,] -0.006468865  0.00539682852
# [2,]  0.003740457 -0.00004601516
# 
# [[4]]
# [1] -0.005880509 -0.002317926

# Check the first few predictions from the initial nnet before training
head(predict_on_batch(object = model, x = X1))
#           [,1]      [,2]
# [1,] 0.4986581 0.5013419
# [2,] 0.4982750 0.5017250
# [3,] 0.4974762 0.5025238
# [4,] 0.4982801 0.5017199
# [5,] 0.4977485 0.5022514
# [6,] 0.4986086 0.5013914

# Evaluate initial model (our model has loss = 0.6934985)
evaluate(object = model, x = X1, y = Y, batch_size = 400)  # 0.6934986

# Run one iteration of gradient descent
train_on_batch(object = model, x = X1, y = Y)

# Check the weights again
get_weights(model)

# [[1]]
#              [,1]          [,2]
# [1,] -0.002025512  0.0087154731
# [2,]  0.002760389  0.0032557009
# [3,]  0.003623125  0.0003690708
# [4,] -0.010618099 -0.0107602663
# 
# [[2]]
# [1] -0.004693955  0.007963512
# 
# [[3]]
#                [,1]        [,2]
# [1,] -0.00490462780 0.003832591
# [2,]  0.00003715511 0.003657288
# 
# [[4]]
# [1] -0.004931618 -0.003266816
