# logistic regression model

# Notes about this model:
# Due to the low number of samples and relatively large number of predictors, we cannot apply logistic regression using every
# feature because the fit method will not converge. (Try it and see). So, we'll pick two features to fit our logistic regression
# model - Contact and AreaCode in order to demonstrate a few important concepts (e.g. one-hot-encoding)

options(scipen=20, digits=4)

# Load packages
library(data.table)
library(stringr)
library(caret)  # for assessing feature importances
library(pROC)

#======================================================================================================
# Load Data (Assumes your current working directory is the Rank Sales Leads problem directory)

train <- fread("_Data/train.csv", select=c("LeadID", "CompanyName", "PhoneNumber", "Contact", "Sale"))
test <- fread("_Data/test.csv", select=c("LeadID", "CompanyName", "PhoneNumber", "Contact", "Sale"))

#======================================================================================================
# Really quick and dirty analysis

# View the raw data
train

# Some thoughts & ideas come to mind
# - Phone number probably isn't useful... but maybe AreaCode is since it should separate the samples by geography => demographics
# - It probably makes sense to order the Contact values: general line < other < manager < owner (this affects the model's construction)

# Now let's see what our overall hit ratio is
train[, list(Samples=.N, Sales=sum(Sale), HitRatio = sum(Sale)/.N)]  # .35

#======================================================================================================
# Feature engineering and transforming the training dataset

# Some things to keep in mind
# - logistic regression only accepts numeric and logical values. So we need to convert Contact to numeric
# - We need to extract AreaCode from PhoneNumber, and then one-hot-encode them (because it doesn't make sense to order AreadCodes)
# - The test set could have an AreaCode not seen in the train set (it doesn't, but it could. So we'll account for that)

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

# Now we need to create a dummy variable for each area code (i.e. one-hot-encoding)
train_dummies <- dcast(train[, list(LeadID, AreaCode)], LeadID ~ AreaCode, fun.aggregate=function(x) 1, fill=0, value.var="AreaCode")
setnames(train_dummies, colnames(train_dummies), paste0("AC", colnames(train_dummies)))  # Prefix each column name with "AC"
setnames(train_dummies, "ACLeadID", "LeadID")  # Change "ACLeadID" back to "LeadID"
train <- train_dummies[train, on="LeadID"]  # Merge the dummy columns back to the train dataset

test_dummies <- dcast(test[, list(LeadID, AreaCode)], LeadID ~ AreaCode, fun.aggregate=function(x) 1, fill=0, value.var="AreaCode")
setnames(test_dummies, colnames(test_dummies), paste0("AC", colnames(test_dummies)))  # Prefix each column name with "AC"
setnames(test_dummies, "ACLeadID", "LeadID")  # Change "ACLeadID" back to "LeadID"
test <- test_dummies[test, on="LeadID"]  # Merge the dummy columns back to the test dataset

# Our test dataset needs to include every dummy variable that's in the train set. Add any missing area code columns with all 0s, if needed.
areacodes_in_train_not_in_test <- setdiff(colnames(train_dummies), colnames(test_dummies))
if(length(areacodes_in_train_not_in_test) > 0) test[, eval(parse(text= paste0("`:=`(", paste0(areacodes_in_train_not_in_test, "=0L", collapse=","), ")") ))]

#======================================================================================================
# Logistic Regression Model

# We have to use the formula syntax here, so something like "Sale ~ Contact + AC310 + ..."
# Also, we need to exclude one of the area codes to avoid a rank-deficient matrix (every column must be independent of the others)
# Note that we could just punch "Sale ~ Contact + AC504 + AC646" into the glm() method, but instead we'll build the formula string dynamically

features <- c("Contact", colnames(train_dummies[, !c("LeadID", "AC310"), with=FALSE]))
formula_str <- paste("Sale ~", paste(features, collapse=" + "))
model <- glm(formula_str, family=binomial(link='logit'), data=train)

#--------------------------------------------------
# Evaluate the fit

summary(model)
varImp(model, scale=FALSE)

#======================================================================================================
# Make some predictions on the test set & evaluate the results

# Make Predictions
test[, ProbSale := predict(model, newdata=test, type='response')]

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
rocCurve$auc  # 0.812
plot(rocCurve)

