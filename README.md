# Machine Learning Problem Bible (MLPB)

MLPB is meant to become an organized collection of machine learning problems and solutions. In practice, machine learning often goes like this

> *I have this problem... I need to classify something as A, B or C using a combination of numeric and categorical features.  If I could find a similar problem, maybe I could modify the solution to work for my needs.*

This is where MLPB steps in. Want to see machine learning problems with sparse data? Got it. Want to compare Scikit-learn’s RandomForestRegressor with R’s randomForest? Got it. Need an example of predicting a ranked target variable? Got it.

## How It Works

MLPB contains a directory of *Problems*. Within each problem is a designated *\_Data* directory and one or more scripts with a solution to the problem. This looks something like

```
Problems/

  Classify Iris Species/
    _Data/
      iris.csv
      train.csv
      test.csv
    predict_species_xgb.R
    
  Predict NFL Game Winner/
    _Data/
      train.csv
      test.csv
    random_forest_model.py
    random_forest_model.R
```

Most of these directories should include a *README.md* file providing details about the problem, data, and solution(s). You can browse all the problems in MLPB's [wiki](https://github.com/ben519/MLPB/wiki). You can also search for problems with specific tags like [mult-class classification], [sparse-data], [NLP], etc.

## Contact
If you'd like to contact me regarding bugs, questions, or general consulting, feel free to drop me a line - bgorman519@gmail.com

## Support
Found this *free* repo helpful? Show your support and [buy some merch](https://shop.gormanalysis.com/)!
[![GormAnalysis Shop](https://www.gormanalysis.com/ads/gormanalysis-shop.jpg)](https://shop.gormanalysis.com/)
