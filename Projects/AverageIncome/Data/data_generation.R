# Load libraries & set random number generator seed for reproducibility

library(data.table)
set.seed(2016)

#======================================================================================================
# Build datasets

countries <- data.table(
  CountryID=1:4, 
  Regions=pmax(1, round(rnorm(n=4, mean=10, sd=10))),
  CountryIncome=rnorm(4, mean=60000, sd=10000)
)

regions <- data.table(
  CountryID=rep(countries$CountryID, countries$Regions),
  RegionID=1:sum(countries$Regions), 
  Cities=pmax(1, round(rnorm(n=sum(countries$Regions), mean=10, sd=10))),
  RegionIncome=rnorm(sum(countries$Regions), mean=rep(countries$CountryIncome, countries$Regions), sd=5000)
)

cities <- data.table(
  RegionID=rep(regions$RegionID, regions$Cities),
  CityID=1:sum(regions$Cities),
  Samples=pmax(1, round(rnorm(n=sum(regions$Cities), mean=1, sd=5))),
  CityIncome=rnorm(sum(regions$Cities), mean=rep(regions$RegionIncome, regions$Cities), sd=5000)
)

samples <- data.table(
  CityID=rep(cities$CityID, cities$Samples),
  SampleID=1:sum(cities$Samples),
  Income=rnorm(sum(cities$Samples), mean=rep(cities$CityIncome, cities$Samples), sd=10000)
)

#--------------------------------------------------
# Merge all the datasets into a flat data.table

dt <- countries[regions, on="CountryID"][cities, on="RegionID"][samples, on="CityID"]
dt[, `:=`(ID=.I, Cities=NULL, Regions=NULL, Samples=NULL, CountryIncome=NULL, RegionIncome=NULL, CityIncome=NULL)]
setcolorder(dt, unique(c("ID", colnames(dt))))

#======================================================================================================
# train-test split:

test <- dt[sample(nrow(dt), 305)]  # use 25% of the data for the test set
train <- dt[!test, on="ID"]  # use the rest for the train set (ha)

#--------------------------------------------------
# Build an additional simple verion of the test set in which every city in test is also seen in train

test_simple <- test[CityID %in% unique(train$CityID)]

#======================================================================================================
# Save files to disc

write.csv(train, "train.csv", row.names=FALSE)
write.csv(test, "test.csv", row.names=FALSE)
write.csv(test_simple, "test_simple.csv", row.names=FALSE)
