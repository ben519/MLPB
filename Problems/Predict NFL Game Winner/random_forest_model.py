# Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#======================================================================================================
# Load Data (Assumes your current working directory is the Predict NFL Game Winner problem directory)

train = pd.read_csv("_Data/train.csv")
test = pd.read_csv("_Data/test.csv")

#======================================================================================================
# Format the training data to the specifications for RandomForestClassifier

#--------------------------------------------------
# Convert categorical features to pandas Category type

train.Opponent = pd.Categorical(train.Opponent)
test.Opponent = pd.Categorical(test.Opponent, categories=pd.unique(pd.concat([train.Opponent, test.Opponent])))

#--------------------------------------------------
# Split the training features from the target. Use pd.get_dummies() to one-hot encode categorical values

train_X = train[["OppRk", "SaintsAtHome", "Expert1PredWin", "Expert2PredWin"]]
train_X = pd.concat([train_X, pd.get_dummies(train.Opponent)], axis=1)

test_X = test[["OppRk", "SaintsAtHome", "Expert1PredWin", "Expert2PredWin"]]
test_X = pd.concat([test_X, pd.get_dummies(test.Opponent)], axis=1)

train_y = train.SaintsWon

#======================================================================================================
# Build some random forest models

#--------------------------------------------------
# Random Forest with 101 trees and the rest defaults

# Train the model
rf = RandomForestClassifier(n_estimators=101)
rf.fit(X=train_X, y=train_y)

# View the 2nd tree  (To plot it, see http://scikit-learn.org/stable/modules/tree.html#tree)
rf.estimators_[1]
rf.estimators_[1].feature_importances_  # feature importances for train_X.columns

# Make predictions on the test set
rf.predict(test_X)  # class predictions
rf.predict_proba(test_X)  # probabilities for rf.classes_

# What would the model predict for each training sample
pd.DataFrame({'Truth': train.SaintsWon, 'Fitted': rf.predict(train_X), 'FittedProbTRUE': rf.predict_proba(train_X)[:, 1]})

#--------------------------------------------------
# Random Forest with 101 trees, testing 3 features at each split

# Train the model
rf = RandomForestClassifier(n_estimators=101, max_features=3)
rf.fit(X=train_X, y=train_y)

# View the importance of each feature. See 
rf.feature_importances_  # importances of train_X.columns
