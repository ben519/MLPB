# gradient boosting model using xgboost

# Notes about this model:
# ...

options(scipen=20, digits=4)

# Load packages
library(data.table)
library(stringr)
library(Matrix)
library(xgboost)
library(pROC)

#======================================================================================================
# Load Data (Assumes your current working directory is the Rank Sales Leads problem directory)

train <- fread("_Data/train.csv")
test <- fread("_Data/test.csv")

#======================================================================================================
# Really quick and dirty analysis

# View the raw data
train

# Some thoughts & ideas come to mind
# - CompanyName could be useful in theory, but we're not going to deal with the complexities of NLP feature engineering here...
# - Phone number probably isn't useful... but maybe AreaCode is since it should separate the samples by geography => demographics
# - It probably makes sense to order the Contact values: general line < other < manager < owner (this affects the model's construction)
# - Website (like PhoneNumber) is too unique to be useful, but maybe businesses that use .net and .org are less likely to buy our software

# Now let's see what our overall hit ratio is
train[, list(Samples=.N, Sales=sum(Sale), HitRatio = sum(Sale)/.N)]  # .35

#======================================================================================================
# Feature engineering and transforming the training dataset

# Some things to keep in mind
# - We have to deal with missing values (NAs). We could impute something into their place, but this is likely to degrade
#   the model's performace. NAs here have special meaning. E.g. FacebookLikes = NA means the company does not have a facebook.
#   This has very different implications than, "The company has a facebook, but we didn't bother to track how many Likes it has"
# - randomForest only accepts numeric and factor values and it can't handle NAs. So we need to convert categorical features to factors
#   and change NAs to some specific value, e.g. "NA_Val", or -1 for numeric features. Also need to convert Sale from logical to factor
# - randomForest can only handle factors with up to 53 unique levels. We won't run into this issue since out dataset is small, but for
#   illustration we'll create a "catch all" group for TypeOfBusiness to use for uncommon business types
# - The test set could have a TypeOfBusiness not seen in the training set. Need to prepare for that!

#--------------------------------------------------
# Create some row indexes to help us populate sparse matricies

train[, RowIdx := .I]
test[, RowIdx := .I]

#--------------------------------------------------
# TypeOfBusiness

# Generate a map to map the values in TypeOfBusiness to a sparse matrix
tobMap <- train[, list(Samples=.N), by=TypeOfBusiness]
tobMap[, TobIdx := .I]

# train
train[tobMap, TobIdx := TobIdx, on="TypeOfBusiness"]
trainTOBSparseM <- sparseMatrix(i=train$RowIdx, j=train$TobIdx, x=1, dimnames=list(NULL, tobMap$TypeOfBusiness))  # build a sparse matrix

# test
test[tobMap, TobIdx := TobIdx, on="TypeOfBusiness"]
testTOBSparseM <- sparseMatrix(i=test[!is.na(TobIdx)]$RowIdx, j=test[!is.na(TobIdx)]$TobIdx, x=1, dims=c(nrow(test), nrow(tobMap)), dimnames=list(NULL, tobMap$TypeOfBusiness))

#--------------------------------------------------
# Contact (convert to numeric, 1-4)

# In this case, we know all the possible contact types
contacts <- c("general line", "other", "manager", "owner")  # Note the order of the elements
train[, Contact := match(Contact, contacts)]
test[, Contact := match(Contact, contacts)]

#--------------------------------------------------
# AreaCode

train[, AreaCode := substr(PhoneNumber, 1, 3)]  # train
test[, AreaCode := substr(PhoneNumber, 1, 3)]  # test

# Generate a map to map the values in AreaCode to a sparse matrix
areacodeMap <- train[, list(Samples=.N), by=AreaCode]
areacodeMap[, AreaCodeIdx := .I]

# train
train[areacodeMap, AreaCodeIdx := AreaCodeIdx, on="AreaCode"]
trainAreaCodeSparseM <- sparseMatrix(i=train$RowIdx, j=train$AreaCodeIdx, x=1, dimnames=list(NULL, areacodeMap$AreaCode))  # build a sparse matrix

# test
test[areacodeMap, AreaCodeIdx := AreaCodeIdx, on="AreaCode"]
testAreaCodeSparseM <- sparseMatrix(i=test[!is.na(AreaCodeIdx)]$RowIdx, j=test[!is.na(AreaCodeIdx)]$AreaCodeIdx, x=1, dims=c(nrow(test), nrow(areacodeMap)), dimnames=list(NULL, areacodeMap$AreaCode))

#--------------------------------------------------
# Website Extension  (note: str_detect comes from the stringr package. Thanks Hadley)

extensions <- c("com", "net", "org", "other", "none")

# train
train[is.na(Website), WebsiteExtension := "none"]
train[str_detect(Website, ".com"), WebsiteExtension := "com"]
train[str_detect(Website, ".net"), WebsiteExtension := "net"]
train[str_detect(Website, ".org"), WebsiteExtension := "org"]
train[is.na(WebsiteExtension), WebsiteExtension := "other"]

# test
test[is.na(Website), WebsiteExtension := "none"]
test[str_detect(Website, ".com"), WebsiteExtension := "com"]
test[str_detect(Website, ".net"), WebsiteExtension := "net"]
test[str_detect(Website, ".org"), WebsiteExtension := "org"]
test[is.na(WebsiteExtension), WebsiteExtension := "other"]

# Generate a map to map the values in WebsiteExtension to a sparse matrix
extensionMap <- train[, list(Samples=.N), by=WebsiteExtension]
extensionMap[, ExtensionIdx := .I]

# train
train[extensionMap, ExtensionIdx := ExtensionIdx, on="WebsiteExtension"]
trainExtensionSparseM <- sparseMatrix(i=train$RowIdx, j=train$ExtensionIdx, x=1, dimnames=list(NULL, extensionMap$WebsiteExtension))  # build a sparse matrix

# test
test[extensionMap, ExtensionIdx := ExtensionIdx, on="WebsiteExtension"]
testExtensionSparseM <- sparseMatrix(i=test[!is.na(ExtensionIdx)]$RowIdx, j=test[!is.na(ExtensionIdx)]$ExtensionIdx, x=1, dims=c(nrow(test), nrow(extensionMap)), dimnames=list(NULL, extensionMap$WebsiteExtension))

#--------------------------------------------------
# FacebookLikes

# Note -1L is type integer vs -1 which is type double
train[is.na(FacebookLikes), FacebookLikes := -1L]  # train
test[is.na(FacebookLikes), FacebookLikes := -1L]  # test

#--------------------------------------------------
# TwitterFollowers

train[is.na(TwitterFollowers), TwitterFollowers := -1L]  # train
test[is.na(TwitterFollowers), TwitterFollowers := -1L]  # test

#--------------------------------------------------
# Sale (the target variable)

# Convert to numeric for xgboost
train[, Sale := Sale*1]

#--------------------------------------------------
# Combine training features into a single sparse matrix

# Insert the non sparse features into a dgCMatrix
trainNonSparseFeats <- Matrix(as.matrix(train[, list(Contact, FacebookLikes, TwitterFollowers)]), sparse=TRUE)
testNonSparseFeats <- Matrix(as.matrix(test[, list(Contact, FacebookLikes, TwitterFollowers)]), sparse=TRUE)

# Combine all the sparse matrices into one big sparse matrix
trainM <- do.call(cBind, list(trainNonSparse, trainTOBSparseM, trainAreaCodeSparseM, trainExtensionSparseM))
testM <- do.call(cBind, list(testNonSparse, testTOBSparseM, testAreaCodeSparseM, testExtensionSparseM))

# Extract the column names of the matrix into a vector called 'features'
features <- c(dimnames(trainM)[[2]])

#======================================================================================================
# XGBoost Model

set.seed(2016)
boostingParams = list(objective="binary:logistic", eta=0.01, max.depth=3, eval_metric="auc", colsample_bytree=.75)
bst <- xgboost(params=boostingParams, data=trainM, label=as.matrix(train$Sale)*1, nrounds=10)

#--------------------------------------------------
# Check the feature importances

xgb.importance(bst, feature_names=features)

#======================================================================================================
# Make some predictions on the test set & evaluate the results

# Make Predictions
test[, ProbSale := predict(bst, newdata=testM)]

#--------------------------------------------------
# Rank the predictions from most likely to least likely

setorder(test, -ProbSale)
test[, ProbSaleRk := .I]

#--------------------------------------------------
# Take a look

test[, list(ProbSaleRk, CompanyName, ProbSale, Sale)]  # Looks perty good!

#--------------------------------------------------
# Let's evaluate the results using area under the ROC curve using the pROC package

rocCurve <- roc(response=test$Sale, predictor=test$ProbSale, direction="<")
rocCurve$auc  # 0.844
plot(rocCurve)

