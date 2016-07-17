# Logistic Regression model

# Notes about this model:
# Due to the low number of samples and relatively large number of predictors, we cannot apply logistic regression using every
# feature because the fit method will not converge. (Try it and see). So, we'll pick two features to fit our logistic regression
# model - Contact and AreaCode in order to demonstrate a few important concepts (e.g. one-hot-encoding)

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Display Settings
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 190)

# # Set working directory
# os.chdir("/Path/To/Rank Sales Leads")

#======================================================================================================
# Load Data (Assumes your current working directory is the Rank Sales Leads problem directory)

train = pd.read_csv("_Data/train.csv", usecols=['LeadID', 'CompanyName', 'PhoneNumber', 'Contact', 'Sale'], dtype={'PhoneNumber':str})
test = pd.read_csv("_Data/test.csv", usecols=['LeadID', 'CompanyName', 'PhoneNumber', 'Contact', 'Sale'], dtype={'PhoneNumber':str})

#======================================================================================================
# Really quick and dirty analysis

# View the raw data
train

# Some thoughts & ideas come to mind
# - Phone number probably isn't useful... but maybe AreaCode is since it should separate the samples by geography => demographics
# - It probably makes sense to order the Contact values: general line < other < manager < owner (this affects the model's construction)

# Now let's see what our overall hit ratio is
train.Sale.sum()/train.shape[0]  # .35

#======================================================================================================
# Feature engineering and transforming the training dataset

# Some things to keep in mind
# - logistic regression only accepts numeric and logical values. So we need to convert Contact to numeric
# - We need to extract AreaCode from PhoneNumber, and then one-hot-encode them (because it doesn't make sense to order AreadCodes)
# - The test set could have an AreaCode not seen in the train set (it doesn't, but it could. So we'll account for that)

#--------------------------------------------------
# AreaCode

# train
train['AreaCode'] = train.PhoneNumber.str[:3]
train['AreaCode'] = pd.Categorical(train.AreaCode)
train_areacode_dummies = pd.get_dummies(train.AreaCode, prefix='AC')
train = pd.concat([train, train_areacode_dummies], axis=1)

# test
test['AreaCode'] = test.PhoneNumber.str[:3]
test['AreaCode'] = pd.Categorical(test.AreaCode, categories=train.AreaCode.cat.categories)
test_areacode_dummies = pd.get_dummies(test.AreaCode, prefix='AC')
test = pd.concat([test, test_areacode_dummies], axis=1)

#--------------------------------------------------
# Contact (convert to numeric, 1-4)

# In this case, we know all the possible contact types
contacts = ["general line", "other", "manager", "owner"]  # Note the order of the elements
train['Contact'] = pd.Categorical(train.Contact, categories=contacts, ordered=True).codes
test['Contact'] = pd.Categorical(test.Contact, categories=contacts, ordered=True).codes

#======================================================================================================
# Logistic Regression Model

features = ['Contact'] + train_areacode_dummies.columns.tolist()

logreg = LogisticRegression(C=1.0)  # Note that C controls the effect regularization
logreg.fit(X=train[features].values, y=train.Sale.values)

#======================================================================================================
# Make some predictions on the test set & evaluate the results

test['ProbSale'] = logreg.predict_proba(test[features].values)[:, 1]

#--------------------------------------------------
# Rank the predictions from most likely to least likely

test.sort_values('ProbSale', inplace=True, ascending=False)
test['ProbSaleRk'] = np.arange(test.shape[0])

#--------------------------------------------------
# Take a look

test[['ProbSaleRk', 'CompanyName', 'ProbSale', 'Sale']]  # Looks perty good!

#--------------------------------------------------
# Evaluate the results using area under the ROC curve

roc_auc_score(y_true=test.Sale, y_score=test.ProbSale)  # 1
