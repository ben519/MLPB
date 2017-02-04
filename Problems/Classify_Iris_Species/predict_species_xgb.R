# Iris example
# Predicting Species from Sepal.Length, Sepal.Width, Petal.Length and Petal.Width

#======================================================================================================
# Load packages

lapply(c("data.table", "ggplot2", "xgboost"), require, character.only=T)

#======================================================================================================
# IMPORTANT: Set your working directory to this problem directoy, "Classify Iris Species"

setwd("Path/To/Classify Iris Species")

#======================================================================================================
# Load data

train <- fread("_Data/train.csv")
test <- fread("_Data/train.csv")

# Convert Species to factor type
train[, Species := factor(Species, levels=c("setosa", "versicolor", "virginica"))]

#======================================================================================================
# Explore

#--------------------------------------------------
# Overall distribution of Species

train[, list(.N, Pcnt=.N/nrow(train)), keyby=Species]

#--------------------------------------------------
# A vs B vs Species

# Sepal.Length vs Sepal.Width
ggplot(train, aes(x=Sepal.Length, y=Sepal.Width, color=Species))+geom_point(size=3)

# Sepal.Length vs Petal.Length
ggplot(train, aes(x=Sepal.Length, y=Petal.Length, color=Species))+geom_point(size=3)

# Sepal.Length vs Petal.Width
ggplot(train, aes(x=Sepal.Length, y=Petal.Width, color=Species))+geom_point(size=3)

# Sepal.Width vs Petal.Length
ggplot(train, aes(x=Sepal.Width, y=Petal.Length, color=Species))+geom_point(size=3)

# Sepal.Width vs Petal.Width
ggplot(train, aes(x=Sepal.Width, y=Petal.Width, color=Species))+geom_point(size=3)

# Petal.Length vs Petal.Width
ggplot(train, aes(x=Petal.Length, y=Petal.Width, color=Species))+geom_point(size=3)


#======================================================================================================
# xgboost

#--------------------------------------------------
# Prep data

# Build matrix of training features
train_X <- as.matrix(train[, !"Species", with=FALSE])

# Build 1-column matrix of training samples (convert labels to 0, 1, ... to satisfy xgboost's required input format)
train_y <- as.matrix(as.integer(train$Species)-1)  # (setosa=0, versicolor=1, virginica=2)

#--------------------------------------------------
# Train model

bst <- xgboost(data=train_X, label=train_y, max.depth=2, eta=1, nround=10, nthread=2, num_class=3, objective="multi:softprob")

#======================================================================================================
# Predict

#--------------------------------------------------
# Pred data

# Build matrix of test features
test_X <- as.matrix(test[, !"Species", with=FALSE])

#--------------------------------------------------
# Get predictions

probas <- predict(bst, test_X)  # predicted probabilities. (Prob(setosa)  = e[1, 4, 7, ...], Prob(versicolor)  = e[2, 5, 8, ...], ...)
preds <- data.table(Setosa=probas[seq(1, nrow(test))*3 - 2], Versicolor=probas[seq(1, nrow(test))*3 - 1], Virginica=probas[seq(1, nrow(test))*3 - 0])
preds[, Prediction := which.max(c(Setosa, Versicolor, Virginica)), by=seq_len(nrow(preds))]
preds[, Prediction := factor(levels(train$Species)[Prediction])]

#--------------------------------------------------
# Compare predictions to true results

results <- cbind(test, preds)
results

#--------------------------------------------------
# Feature importance

importance_matrix <- xgb.importance(model = bst, colnames(train_X))
importance_matrix
