---
title: "Machine Learning Project - Weight Lifting Exercise Dataset"
date: "Tuesday, March 17, 2015"
output: html_document
---

The dataset for this project is the weight lifting exrcise datset, which includes measurements from wearable sensors, taken during weight lifting. The goal was to investigate how well weight lifting was performed, and classify it as "correct", or done in one of the common mistaken ways. In this project we use machine learning to create a model that can classify the observation into one of five classes, including the "correct"" class("A") or one of the other classes that represent wrong weight lifting ways.

I downloaded the training set, and read it into R.


```r
pml.train <- read.csv("pml-training.csv")
dim(pml.train)
```

```
## [1] 19622   160
```

The dataset has 160 variables, but many columns are mostly NA or empty values. In addition, there are columns that are not relevant for classification, such as observation number, various time stamps, user names and window numbers. I removed all these variables. The resulting dataset has 53 variables (including the outcome variable, classe)


```r
features <- c(8:11,37:49,60:68,84:86,102, 113:124, 140, 151:160)
pml.train <- pml.train[,features]
dim(pml.train)
```

```
## [1] 19622    53
```

I created the model using the random forest algorithm. I used the train method from the caret package, and did cross-validation with 5 folds. Training the model on my computer took a long time, and the number of folds strongly affects the time it takes, so I did not try a larger number of folds. However, 5-fold cross validation should give a good estimate of the out of sample error. The model was created with classe as the outcome and all other variables as predictors.


```r
library(caret)
set.seed(8181)
fit <- train(classe ~ ., data=pml.train, method="rf", trControl = trainControl(method="cv", number=5))
```
Printing the model shows that indeed the model was created with random forest, and was validated with 5-fold cross validation. 

```r
fit
```

```
## Random Forest 
##
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 

## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 15698, 15698, 15698, 15696, 15698 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9939353  0.9923282  0.001952882  0.002470692
##   27    0.9942920  0.9927797  0.001307501  0.001654223
##   52    0.9886349  0.9856232  0.003093911  0.003914727
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27. 
```
Printing the final model shows that the estimate of the out of sample error is 0.42%.

```r
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.42%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5576    2    1    0    1 0.0007168459
## B   20 3773    4    0    0 0.0063207796
## C    0    9 3401   12    0 0.0061367621
## D    0    0   21 3193    2 0.0071517413
## E    0    1    3    6 3597 0.0027723870
```
Now let's apply the model to the test set:


```r
pml.test <- read.csv("pml-testing.csv")
predict(fit, pml.test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
These predictions proved out to be the correct results (when tested with the project submission script).

