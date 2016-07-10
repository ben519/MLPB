# Predicting Football Game Winner
Predict whether the New Orleans Saints will win a football game against their opponent using fictitious data on historical Saints games.

#### Notes
This problem is particularly challenging because of the hierachical structure of the data: City < Region < Country. For a random person, it makes sense to predict his income close to the income of other people in his city.  But if few or no training samples of people in his city exist, you will have to put more weight on the income level of other people in his region (and so on).

#### Models
At this time, two [XGBoost](https://github.com/dmlc/xgboost) models are given:

 - **predict_income_xgb1.R** - uses average income per City, Region, and Country to train an xgboost model
 - **predict_income_xgb2.R** - one-hot-encodes City, Region, and Country, generating a sparse matrix to train an xgboost model

### Tags
[supervised-learning] [average_income-dataset] [regression] [xgboost] [gradient_boosting] [multi-level-data] [hierarchical-data] [one-hot-encoding]

### References
This problem dataset was inspired by [this question](http://stats.stackexchange.com/questions/221358/how-to-deal-with-hierarchical-nested-data-in-machine-learning) on [CrossValidated](http://stats.stackexchange.com/).