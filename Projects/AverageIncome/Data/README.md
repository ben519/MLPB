# Average Income

#### Description
This dataset includes fictitious income levels (i.e. salaries) of people identified by hierarchical geographic attributes:

**CityID** < **RegionID** < **CountryID**

It mimics a common issue involving multi-level data - if you want to predict the average income of an individual it's best to use the average income of people in his city.  However, if there is insufficient data in his city, one must fall back to using average income in his region (and so on, up the hierarchy). An optimal model will weight each level of the hierarchy, but this poses a number of issues for common machine learning algorithms.

Note that two versions of the test set are given:

* *test.csv* includes CityIDs not seen in the train.csv
* *test_simple.csv* (a subset of test) only has CityIDs which are seen in the train.csv

Also note that the R code used to generate the data is provided via *data_generation.R*

#### Inspiration
This problem dataset was inspired by [this question](http://stats.stackexchange.com/questions/221358/how-to-deal-with-hierarchical-nested-data-in-machine-learning) on [CrossValidated](http://stats.stackexchange.com/).