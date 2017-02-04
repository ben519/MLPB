# Gradient Boosting model using XGBoost

# Notes about this model:
# XGBoost requires a single matrix of numeric values as input. This means non ordered categorical features need to be
# one-hot-encoded. This can result in a really large sparse matrix, but XGBoost allows for the input matrix to be represented
# in compressed format via dgCMatrix, so we'll utilize that option here

# Imports
import os
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Display Settings
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 190)

# # Set working directory
# os.chdir("/Path/To/Rank Sales Leads")

#======================================================================================================
# Load Data (Assumes your current working directory is the Rank Sales Leads problem directory)

train = pd.read_csv("_Data/train.csv", dtype={'PhoneNumber':str})
test = pd.read_csv("_Data/test.csv", dtype={'PhoneNumber':str})

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
train.Sale.sum()/train.shape[0]  # .35

#======================================================================================================
# Feature engineering and transforming the training dataset

# Some things to keep in mind
# - Need to build a sparse matrix for each non-ordered categorical feature: TypeOfBusiness, AreaCode, and WebsiteExtension
# - Need to concatenate all features into one big sparse matrix
# - XGBoost has smart handling for NA values, so we don't need to impute values for NA in FacebookLikes and TwitterFollowers

#--------------------------------------------------
# TypeOfBusiness

# Convert NAs to "NA_Val"
train.TypeOfBusiness.fillna("NA_Val", inplace=True)
test.TypeOfBusiness.fillna("NA_Val", inplace=True)

# Generate a map to map the values in TypeOfBusiness to a sparse matrix
tobMap = pd.DataFrame({'TypeOfBusiness': train.TypeOfBusiness.unique()})
tobMap['TOBColIdx'] = np.arange(tobMap.shape[0])

# Build sparse matrix of train values
train = train.merge(tobMap, on='TypeOfBusiness', how='left')
trainTOBSparseM = sparse.csr_matrix((np.repeat(1, train.shape[0]), (train.index.values, train.TOBColIdx.values)))

# Build sparse matrix of test values
test = test.merge(tobMap, on='TypeOfBusiness', how='left')
testTOBSparseM = sparse.csr_matrix((np.repeat(1, sum(test.TOBColIdx.notnull())), (test[test.TOBColIdx.notnull()].index.values, test[test.TOBColIdx.notnull()].TOBColIdx.values)), [test.shape[0], tobMap.shape[0]])

#--------------------------------------------------
# AreaCode

train['AreaCode'] = train.PhoneNumber.str[:3]
test['AreaCode'] = test.PhoneNumber.str[:3]

# Generate a map to map the values in AreaCode to a sparse matrix
areacodeMap = pd.DataFrame({'AreaCode': train.AreaCode.unique()})
areacodeMap['AreaCodeColIdx'] = np.arange(areacodeMap.shape[0])

# Build sparse matrix of train values
train = train.merge(areacodeMap, on='AreaCode', how='left')
trainAreaCodeSparseM = sparse.csr_matrix((np.repeat(1, train.shape[0]), (train.index.values, train.AreaCodeColIdx.values)))

# Build sparse matrix of test values
test = test.merge(areacodeMap, on='AreaCode', how='left')
testAreaCodeSparseM = sparse.csr_matrix((np.repeat(1, sum(test.AreaCodeColIdx.notnull())), (test[test.AreaCodeColIdx.notnull()].index.values, test[test.AreaCodeColIdx.notnull()].AreaCodeColIdx.values)), [test.shape[0], areacodeMap.shape[0]])

#--------------------------------------------------
# Website Extension

# train
train.loc[train.Website.isnull(), 'WebsiteExtension'] = 'none'
train.loc[train.Website.str.contains('com').replace(np.nan, False), 'WebsiteExtension'] = 'com'
train.loc[train.Website.str.contains('org').replace(np.nan, False), 'WebsiteExtension'] = 'org'
train.loc[train.Website.str.contains('net').replace(np.nan, False), 'WebsiteExtension'] = 'net'
train.loc[train.WebsiteExtension.isnull(), 'WebsiteExtension'] = 'other'

# test
test.loc[test.Website.isnull(), 'WebsiteExtension'] = 'none'
test.loc[test.Website.str.contains('com').replace(np.nan, False), 'WebsiteExtension'] = 'com'
test.loc[test.Website.str.contains('org').replace(np.nan, False), 'WebsiteExtension'] = 'org'
test.loc[test.Website.str.contains('net').replace(np.nan, False), 'WebsiteExtension'] = 'net'
test.loc[test.WebsiteExtension.isnull(), 'WebsiteExtension'] = 'other'

# Generate a map to map the values in WebsiteExtension to a sparse matrix
extensionMap = pd.DataFrame({'WebsiteExtension': train.WebsiteExtension.unique()})
extensionMap['ExtensionColIdx'] = np.arange(extensionMap.shape[0])

# Build sparse matrix of train values
train = train.merge(extensionMap, on='WebsiteExtension', how='left')
trainExtensionSparseM = sparse.csr_matrix((np.repeat(1, train.shape[0]), (train.index.values, train.ExtensionColIdx.values)))

# Build sparse matrix of test values
test = test.merge(extensionMap, on='WebsiteExtension', how='left')
testExtensionSparseM = sparse.csr_matrix((np.repeat(1, sum(test.ExtensionColIdx.notnull())), (test[test.ExtensionColIdx.notnull()].index.values, test[test.ExtensionColIdx.notnull()].ExtensionColIdx.values)), [test.shape[0], extensionMap.shape[0]])

#--------------------------------------------------
# Contact (convert to numeric, 1-4)

# In this case, we know all the possible contact types
contacts = ["general line", "other", "manager", "owner"]  # Note the order of the elements
train['Contact'] = pd.Categorical(train.Contact, categories=contacts, ordered=True).codes
test['Contact'] = pd.Categorical(test.Contact, categories=contacts, ordered=True).codes

#--------------------------------------------------
# Combine training features into a single sparse matrix and then convert to xgb.DMatrix type

# Insert the non sparse features into a sparse matrix
trainNonSparseFeats = sparse.csr_matrix(train[['Contact', 'FacebookLikes', 'TwitterFollowers']])
testNonSparseFeats = sparse.csr_matrix(test[['Contact', 'FacebookLikes', 'TwitterFollowers']])

# Concatenate all sparse matrices
trainM = sparse.hstack([trainNonSparseFeats, trainTOBSparseM, trainAreaCodeSparseM, trainExtensionSparseM])
testM = sparse.hstack([testNonSparseFeats, testTOBSparseM, testAreaCodeSparseM, testExtensionSparseM])

# Convert to type DMatrix for xgboost
trainM = xgb.DMatrix(data=trainM, label=train.Sale)
testM = xgb.DMatrix(data=testM, label=test.Sale)

#======================================================================================================
# XGBoost Model

np.random.seed(2016)  # eta=.3, max.depth=10, subsample=.75, colsample_bytree=.75, min_child_weight=1, gamma=0, lambda=0, alpha=0
boosting_params = {'bst:eta':0.3, 'bst:max_depth':10, 'bst:subsample':.75, 'bst:colsample_bytree':.75, 'min_child_weight':1, 'gamma':0, 'lambda':0, 'alpha':0, 'objective':'binary:logistic', 'eval_metric':'auc'}
bst = xgb.train(params=boosting_params, dtrain=trainM, num_boost_round=10)

#======================================================================================================
# Make some predictions on the test set & evaluate the results

test['ProbSale'] = bst.predict(testM)

#--------------------------------------------------
# Rank the predictions from most likely to least likely

test.sort_values('ProbSale', inplace=True, ascending=False)
test['ProbSaleRk'] = np.arange(test.shape[0])

#--------------------------------------------------
# Take a look

test[['ProbSaleRk', 'CompanyName', 'ProbSale', 'Sale']]  # Looks perty good!

#--------------------------------------------------
# Evaluate the results using area under the ROC curve

roc_auc_score(y_true=test.Sale, y_score=test.ProbSale)  # 0.875
