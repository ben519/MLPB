options(scipen=20, digits=4)

# Load packages
library(randomForest)  # version 4.6-12

#======================================================================================================
# Load Data (Assumes your current working directory is the Football project directory)

train <- read.csv("Data/train.csv")
test <- read.csv("Data/test.csv")

#======================================================================================================
# Format the training data to the specifications for randomForest
# See help(randomForest)

#--------------------------------------------------
# Convert boolean features to factors (i.e. categorical features with levels {TRUE, FALSE}. You could also
# convert to intgers {0, 1})

train$SaintsAtHome <- factor(train$SaintsAtHome, levels=c("FALSE", "TRUE"))
train$Expert1PredWin <- factor(train$Expert1PredWin, levels=c("FALSE", "TRUE"))
train$Expert2PredWin <- factor(train$Expert2PredWin, levels=c("FALSE", "TRUE"))
train$SaintsWon <- factor(train$SaintsWon, levels=c("FALSE", "TRUE"))
test$SaintsAtHome <- factor(test$SaintsAtHome, levels=c("FALSE", "TRUE"))
test$Expert1PredWin <- factor(test$Expert1PredWin, levels=c("FALSE", "TRUE"))
test$Expert2PredWin <- factor(test$Expert2PredWin, levels=c("FALSE", "TRUE"))

#--------------------------------------------------
# Split the training features from the target

train_X <- train[, c("Opponent", "OppRk", "SaintsAtHome", "Expert1PredWin", "Expert2PredWin")]
test_X <- test[, c("Opponent", "OppRk", "SaintsAtHome", "Expert1PredWin", "Expert2PredWin")]
test_X$Opponent <- factor(test_X$Opponent, levels=levels(train_X$Opponent))  # Make sure the levels of Opponent in test are the same as train
train_y <- train$SaintsWon

#======================================================================================================
# Build some random forest models

#--------------------------------------------------
# Random Forest with 101 trees and the rest defaults

# Train the model
rf <- randomForest(x=train_X, y=train_y, ntree=101)

# View results. Confusion matrix based on out of bag samples applied to each tree
rf

# View the 2nd tree
getTree(rf, k=2, labelVar=TRUE)  # see help(getTree) for details

# Make predictions on the test set
predict(rf, test_X)  # class predictions
predict(rf, test_X, type="prob")  # probabilities

# What would the model predict for each training sample
data.frame(Truth=train$SaintsWon, Fitted=predict(rf, train_X), FittedProbTRUE=predict(rf, train_X, type="prob")[, 2])

#--------------------------------------------------
# Random Forest with 101 trees, testing 3 features at each split and keeping track of variable importance

# Train the model
rf <- randomForest(x=train_X, y=train_y, ntree=101, mtry=3, importance=TRUE)

# View the importance of each feature. See help(importance.randomForest)
importance(rf, type=1)  # type=1 -> mean decrease in accuracy
importance(rf, type=2)  # type=2 -> mean decrease in node impurity

#--------------------------------------------------
# Browse more properties of the model

str(rf)
