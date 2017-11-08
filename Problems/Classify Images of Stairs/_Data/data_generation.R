# Code to generate train & test datasets for the "Classify Images of Stairs" problem

#======================================================================================================
# settings

options(scipen = 20)

#--------------------------------------------------
# packages

library(data.table)

#--------------------------------------------------
# working directory

# setwd("~/Projects/R/MLPB/Problems/Classify Images of Stairs")

if(basename(getwd()) != "Classify Images of Stairs")
  stop("This script expects your current working directory to be 'Classify Images of Stairs/'. Either setwd() accordingly, 
       or delete this condition and fix the frwite(dataset, file = '_Data/dataset.csv') path in the code below.")

#======================================================================================================
# Datasets

# Build datasets:
# train, test

set.seed(2017)

stairsWide <- rbind(
  data.table(
    R1C1 = sample(0:10, size = 125, replace = TRUE),
    R1C2 = sample(150:255, size = 125, replace = TRUE),
    R2C1 = sample(150:255, size = 125, replace = TRUE),
    R2C2 = sample(150:255, size = 125, replace = TRUE)
  ),
  data.table(
    R1C1 = sample(150:255, size = 125, replace = TRUE),
    R1C2 = sample(0:10, size = 125, replace = TRUE),
    R2C1 = sample(150:255, size = 125, replace = TRUE),
    R2C2 = sample(150:255, size = 125, replace = TRUE)
  )
)
stairsWide[, IsStairs := TRUE]

nonStairsWide <- data.table(
  R1C1 = sample(x=0:255, size = 250, replace = TRUE),
  R1C2 = sample(x=0:255, size = 250, replace = TRUE),
  R2C1 = sample(x=0:255, size = 250, replace = TRUE),
  R2C2 = sample(x=0:255, size = 250, replace = TRUE)
)
nonStairsWide[, IsStairs := FALSE]

# combine stairsWide and nonStairsWide into 'train'
train <- rbind(stairsWide, nonStairsWide)
train <- train[sample(.N, .N)]  # random shuffle
train[, ImageId := .I]
setcolorder(train, c("ImageId", "R1C1", "R1C2", "R2C1", "R2C2", "IsStairs"))

# split train into train/test
test <- train[sample(.N, 100)][order(ImageId)]
train <- train[!test, on="ImageId"][order(ImageId)]

#======================================================================================================
# Save datasets

fwrite(test, file = "_Data/test.csv")
fwrite(train, file = "_Data/train.csv")
