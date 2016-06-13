# Machine Learning Problem Bible (MLPB)

MLPB is meant to become an organized collection of machine learning problems and solutions. In practice, machine learning often goes like this

> *I have this problem... I need to classify something as A, B or C using a combination of numeric and categorical features.  If I could find a similar problem, maybe I could modify the solution to work for my needs.*

This is where MLPB steps in. Want to see ML problems with sparse data? Got it. Want to compare Scikit-learn’s RandomForestRegressor with R’s randomForest? No problem. Need an example of predicting a ranked target variable? You’ve come to the right place.

## How It Works

MLPB contains a directory of *Datasets* and a directory of *Problems*. For example

```
DataSets/
  Iris/
    Description.md
    iris.csv

Problems/
  Predicting Iris Species/
    Description.md
    predict_species_rf.py
    predict_species_rf.R

  Predicting Iris Sepal Length/
    Description.md
    predict_sepal_length_rf.py
    predict_sepal_length_rf.R
    iris_xgboost.py
```

Each *Description.md* file includes the details about the dataset/problem, and files like *predict_species_rf.py* and *predict_species_rf.R* provide example solutions (in this case using random forest) to their respective problem.

On top of all this, MLPB should (hopefully) develop a well documented wiki making it easy to search for problems using  characteristics like "mult-class classification", "sparse data", "unbalanced target", etc.
