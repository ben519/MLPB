# Income example
# Predicting Income per person using City, Region, Country
# The objective of this model will be to minimize Root Mean Square Error

# This model will one-hot-encode each CityID, RegionID, and CountryID into 3 separate sparse matrices, 
# concatenate them into 1 big sparse matrix and use those features to train an xgboost regression model

options(scipen=20)

#======================================================================================================
# Load packages

lapply(c("data.table", "ggplot2", "xgboost", "DiagrammeR", "Matrix"), require, character.only=T)

#======================================================================================================
# Helper Functions

# Root Mean Squared Error
rmse <- function(preds, actuals) sqrt(mean((preds-actuals)^2))

#======================================================================================================
# Load data (Assumes your current working directory is the Predict Income problem directory)

train <- fread("_Data/train.csv")
test <- fread("_Data/test.csv")
setnames(test, "Income", "IncomeTruth")

#======================================================================================================
# Build modified training dataset

dt <- rbind(train[, list(CountryID, RegionID, CityID, Income)], test[, list(CountryID, RegionID, CityID, Income=NA)])

countries <- dt[, list(.N, AvgIncome=mean(Income, na.rm=TRUE)), key=CountryID]
countries[, CountryIdx := .I]
regions <- dt[, list(.N, AvgIncome=mean(Income), na.rm=TRUE), key=RegionID]
regions[, RegionIdx := .I]
cities <- dt[, list(.N, AvgIncome=mean(Income), na.rm=TRUE), key=CityID]
cities[, CityIdx := .I]

train[countries, CountryIdx := CountryIdx, on="CountryID"]
train[regions, RegionIdx := RegionIdx, on="RegionID"]
train[cities, CityIdx := CityIdx, on="CityID"]

test[countries, CountryIdx := CountryIdx, on="CountryID"]
test[regions, RegionIdx := RegionIdx, on="RegionID"]
test[cities, CityIdx := CityIdx, on="CityID"]

train[, Idx := .I]
test[, Idx := .I]

train_countryM <- sparseMatrix(i=train$Idx, j=train$CountryIdx, x=1)
train_regionM <- sparseMatrix(i=train$Idx, j=train$RegionIdx, x=1)
train_cityM <- sparseMatrix(i=train$Idx, j=train$CityIdx, x=1)

test_countryM <- sparseMatrix(i=test$Idx, j=test$CountryIdx, x=1)
test_regionM <- sparseMatrix(i=test$Idx, j=test$RegionIdx, x=1)
test_cityM <- sparseMatrix(i=test$Idx, j=test$CityIdx, x=1)

trainM <- do.call(cBind, list(train_countryM, train_regionM, train_cityM))
testM <- do.call(cBind, list(test_countryM, test_regionM, test_cityM))

#======================================================================================================
# xgboost that puppy

features <- c(paste0("Country", countries$CountryID), paste0("Region", regions$RegionID), paste0("City", cities$CityID))

#--------------------------------------------------
# Train model

paramList <- list(eta=.2, gamma=0, max.depth=3, min_child_weight=1, subsample=.9, colsample_bytree=1)  # Test various hyperparameters and values here and see what works best. (A poor man's grid search)
bst.cv <- xgb.cv(params=paramList, data=trainM, label=as.matrix(train$Income), nfold=5, early.stop.round=10, eval_metric="rmse", nrounds=200, prediction=TRUE)
bst <- xgboost(params=paramList, data=trainM, label=as.matrix(train$Income), nrounds=nrow(bst.cv[["dt"]])-10)

#======================================================================================================
# Predict & Evaluate

#--------------------------------------------------
# Predict

train[, IncomeXGB := predict(bst, trainM)]
test[, IncomeXGB := predict(bst, testM)]

#--------------------------------------------------
# Trees

bst.trees <- xgb.model.dt.tree(features, model=bst)
bst.trees[Tree==0]

#--------------------------------------------------
# Importance

xgb.importance(model=bst, features)

#--------------------------------------------------
# Plot a minimal models (3 rounds of training)

bst.minimal <- xgboost(params=paramList, data=trainM, label=as.matrix(train$Income), nrounds=3)
xgb.plot.tree(features, model=bst.minimal)

#--------------------------------------------------
# Evaluate

rmse(train$IncomeXGB, train$Income)  # 9881.507
rmse(test$IncomeXGB, test$IncomeTruth)  # 11270.89

# Errors
train[, SE := (IncomeXGB-Income)^2]
test[, SE := (IncomeXGB-IncomeTruth)^2]
test[order(SE)]
