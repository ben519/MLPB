# Machine Learning Problem Bible (MLPB)

MLPB is meant to become an organized collection of machine learning problems and solutions. In practice, machine learning often goes like this

> *I have this problem... I need to classify something as A, B or C using a combination of numeric and categorical features.  If I could find a similar problem, maybe I could modify the solution to work for my needs.*

This is where MLPB steps in. Want to see ML problems with sparse data? Got it. Want to compare Scikit-learn’s RandomForestRegressor with R’s randomForest? No problem. Need an example of predicting a ranked target variable? You’ve come to the right place.

## How It Works

MLPB contains a directory of *Projects*. Within each project is a designated *Data* directory and one or more machine learning problem directories. This looks something like

```
Projects/

  Iris/
    Data/
      iris.csv
    PredictSpecies/
      predict_species_rf.R
      predict_species_rf.py
      predict_species_xgb.R
    PredictSepalLength/
      iris_sepal_length_xgb.R
      
  JobTitles/
    Data/
      jobtitles.csv
    PredictJobCategory/
      naive_bayes_model.py
      naive_bayes_model.R
```

Most of these directories should include a *README.md* file providing details about the data and/or problem. A single dataset can have have multiple problems, and a single problem can have multiple example models/solutions.

You can browse all the datasets and problems in MLPB's [wiki](https://github.com/ben519/MLPB/wiki). There, you can also search for problems with specific tags like [mult-class classification], [sparse-data], [NLP], etc.
