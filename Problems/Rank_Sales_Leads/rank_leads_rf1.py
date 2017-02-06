# Random Forest model using RandomForestClassifier from scikit-learn

# Notes about this model (as of 2016-07-18):
# Scikit-learn's RandomForestClassifier currently has a deficiency -  unordered categorical values must be one-hot-encoded,
# resulting in a wide sparse training dataset where randomly selected subsets of columns will often contain all 0 values

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# - We have to deal with missing values (NaNs). We could impute mean or median into their place, but this is likely to degrade
#   the model's performace. NaNs here have special meaning. E.g. FacebookLikes = NaN means the company does not have a facebook.
#   This has very different implications than, "The company has a facebook, but we didn't bother to track how many Likes it has"
# - RandomForestClassifier can't handle categorical data, so we need to convert ordered categories to numeric and one-hot-encode
#   unordered categorical fields (keeping in mind that a new category might arise in the test set)

#--------------------------------------------------
# TypeOfBusiness

# Convert NaN to "NA_Val"
train.TypeOfBusiness.fillna('NA_Val', inplace=True)
test.TypeOfBusiness.fillna('NA_Val', inplace=True)

# To help avoid overfitting, and to reduce the number of columns generated from one-hot-encoding,
# we will mark uncommon business types as "other" (freq <= 1)
tobMap = train.groupby('TypeOfBusiness')['TypeOfBusiness'].agg({'count'}).reset_index()
tobMap['TOBGroup'] = tobMap.TypeOfBusiness
tobMap.loc[tobMap['count'] <= 1, 'TOBGroup'] = 'other'
train = train.merge(tobMap.drop('count', axis=1), on='TypeOfBusiness', how='left')
test = test.merge(tobMap.drop('count', axis=1), on='TypeOfBusiness', how='left')

# one-hot-encode
tob_groups = tobMap.TOBGroup.unique()
train['TOBGroup'] = pd.Categorical(train.TOBGroup, categories=tob_groups)
train_tob_dummies = pd.get_dummies(train.TOBGroup, prefix='TOB')
train = pd.concat([train, train_tob_dummies], axis=1)
test['TOBGroup'] = pd.Categorical(test.TOBGroup, categories=tob_groups)
test_tob_dummies = pd.get_dummies(test.TOBGroup, prefix='TOB')
test = pd.concat([test, test_tob_dummies], axis=1)

#--------------------------------------------------
# Website Extension

extensions = ['none', 'com', 'org', 'net', 'other']

# train
train.loc[train.Website.isnull(), 'WebsiteExtension'] = 'none'
train.loc[train.Website.str.contains('com').replace(np.nan, False), 'WebsiteExtension'] = 'com'
train.loc[train.Website.str.contains('org').replace(np.nan, False), 'WebsiteExtension'] = 'org'
train.loc[train.Website.str.contains('net').replace(np.nan, False), 'WebsiteExtension'] = 'net'
train.loc[train.WebsiteExtension.isnull(), 'WebsiteExtension'] = 'other'
train['WebsiteExtension'] = pd.Categorical(train.WebsiteExtension, categories=extensions)
train_extension_dummies = pd.get_dummies(train.WebsiteExtension, prefix='EX')
train = pd.concat([train, train_extension_dummies], axis=1)

# test
test.loc[test.Website.isnull(), 'WebsiteExtension'] = 'none'
test.loc[test.Website.str.contains('com').replace(np.nan, False), 'WebsiteExtension'] = 'com'
test.loc[test.Website.str.contains('org').replace(np.nan, False), 'WebsiteExtension'] = 'org'
test.loc[test.Website.str.contains('net').replace(np.nan, False), 'WebsiteExtension'] = 'net'
test.loc[test.WebsiteExtension.isnull(), 'WebsiteExtension'] = 'other'
test['WebsiteExtension'] = pd.Categorical(test.WebsiteExtension, categories=extensions)
test_extension_dummies = pd.get_dummies(test.WebsiteExtension, prefix='EX')
test = pd.concat([test, test_extension_dummies], axis=1)

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

#--------------------------------------------------
# FacebookLikes

train.FacebookLikes.fillna(-1, inplace=True)
test.FacebookLikes.fillna(-1, inplace=True)

#--------------------------------------------------
# TwitterFollowers

train.TwitterFollowers.fillna(-1, inplace=True)
test.TwitterFollowers.fillna(-1, inplace=True)

#======================================================================================================
# Random Forest Model

features = ['Contact', 'FacebookLikes', 'TwitterFollowers'] + train_tob_dummies.columns.tolist() + train_areacode_dummies.columns.tolist() + test_areacode_dummies.columns.tolist()
rf = RandomForestClassifier(n_estimators=200, max_features=.33, min_samples_leaf=3, random_state=2016)
rf.fit(X=train[features].values, y=train.Sale.values)

#--------------------------------------------------
# Check the importance of features

impotances = pd.DataFrame({'Feature':features, 'Importance':rf.feature_importances_})
impotances.sort_values('Importance', ascending=False)

#======================================================================================================
# Make some predictions on the test set & evaluate the results

test['ProbSale'] = rf.predict_proba(test[features].values)[:,1]

#--------------------------------------------------
# Rank the predictions from most likely to least likely

test.sort_values('ProbSale', inplace=True, ascending=False)
test['ProbSaleRk'] = np.arange(test.shape[0])

#--------------------------------------------------
# Take a look

test[['ProbSaleRk', 'CompanyName', 'ProbSale', 'Sale']]  # Looks perty good!

#--------------------------------------------------
# Evaluate the results using area under the ROC curve

roc_auc_score(y_true=test.Sale, y_score=test.ProbSale)  # 0.75
