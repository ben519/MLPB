options(scipen=20, digits=4)

# Load packages
library(e1071)  # for naiveBayes()
library(tm)  # for counting word frequencies

# ==============================================================================
# Load Data

jobtitles <- read.csv("Datasets/JobTitles/jobtitles.csv", na.strings=c("NA", ""))


# ==============================================================================
# Count word frequencies
# Here we make use of the tm (text mining) package

# first build a Vector Corpus object
my.corpus <- VCorpus(VectorSource(jobtitles$job_title))

# now build a document term matrix
dtm <- DocumentTermMatrix(my.corpus)

# inspect the results
inspect(dtm)


# ==============================================================================
# Train a naive bayes model

# put word frequencies into a data.frame and convert column types from numeric to factor 
# (so naiveBaues() knows the xi is a Bernoulli random veriable and not Gaussian)

# prepare training data
train.x <- data.frame(inspect(dtm)[1:10,]) # use the first 10 samples to build the training set
train.x[, 1:10] <- lapply(train.x, FUN=function(x) factor(x, levels=c("0", "1"))) # convert columns to factor type so that naiveBayes knows features are Bernoulli random variables
train.y <- factor(categories[1:10])

# prepare test data
test.x <- data.frame(inspect(dtm)[11:12,]) # use the last 2 samples to build the test set
test.x[, 1:10] <- lapply(test.x, FUN=function(x) factor(x, levels=c("0", "1"))) # convert columns to factor type

# train model
classifier <- naiveBayes(x=train.x, y=train.y, laplace=0.000000001)  # use laplace (i.e. alpha) of nearly 0

# make prediction on the unlabeled data
predict(classifier, test.x, type="raw")


# ==============================================================================
# Train a naive bayes model with laplace (alpha) = 1

# train model
classifier <- naiveBayes(x=train.x, y=train.y, laplace=1)  # use laplace (i.e. alpha) of 1

# make prediction on the unlabeled data
predict(classifier, test.x, type="raw")
