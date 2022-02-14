library(caret)
library(data.table)
library(DataComputing)
library(Metrics)
library(ggplot2)
library(xgboost)
library(tidyverse)
library(lattice)
library(Rtsne)
library(Metrics)
library(cluster)
library(ClusterR)
library(mlbench)
library(randomForest)
library(e1071)
library(neuralnet)


############# Chapter 8 Machine Learning With the Caret Package


train <- fread("Desktop/R Work/Learning/Intro to ML With R/Data/titanic/train.csv")
str(train)

#2 missing values
table(train$Embarked)

#Not for Sex
table(train$Sex)

#Checking for NA values
apply(train, 2, function(x) any(is.na(x)))

missingAgeIndex <- which(is.na(train$Age), arr.ind=TRUE)
newData <- train[-missingAgeIndex,]
apply(newData, 2, function(x) any(is.na(x)))

#Changing missing value to s 
train$Embarked[train$Embarked == ""] <- "S"

#How much percentage of the train$Age is na [2] for true na and [1] for false na
table(is.na(train$Age))[2]/table(is.na(train$Age))[1]

summary(train$Age)

#Replace na with the  most common age or median but first fill a label if the age
#data is missing 
train$is_age_missing <- ifelse(is.na(train$Age), 1, 0)

#Consolidate people that have sibling and spouses & parent and children 
#So the total of family members traveling
train$travelers <- train$SibSp + train$Parch + 1

#Factorizing data
train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)
train$is_age_missing <- as.factor(train$is_age_missing)

#We selected the desired features
train2 <- subset(train, select = c(Survived, Pclass, Sex, Age,
                                   SibSp, Parch, Fare, Embarked, is_age_missing, travelers))

train2

#Imputation

library(caret)

#Dummies for character and factor columns are affected
#Data in numeric format better from a modeling standpoint
dummy <- dummyVars(~., data = train2[, -1])
dummy
summary(dummy)
dummy_train <- predict(dummy, train2[, -1])
head(dummy_train)


pre.process <- preProcess(dummy_train, method = "bagImpute")
pre.process

#Removes NA values models the values for those variables, pred via bag ^
imputed.data <- predict(pre.process, dummy_train)
head(imputed.data)

#Placing the preProcess Age values back to the original data
train$Age <- imputed.data[, 6]
train2$Age <- imputed.data[,6]
head(train$Age, 20)
head(train2$Age, 20)

#Set train and test
set.seed(123)
partition_indexes <- createDataPartition(train2$Survived, times = 1,
                                         p = 0.7, list = FALSE)
titanic.train <- train2[partition_indexes, ]
titanic.test <- train2[-partition_indexes, ]

##Example of a caret training, trControl is how you want to do the cross validation
##TuneGrid is df of parameter that you want to pass to your model 
train.model <- train(Survived ~ ., data = titanic.train, method = "xgbTree",
                     tuneGrid = tune.grid, trControl = train.control)

#Self explanatory (Pretty Useful) 
getModelInfo("xgbTree")


#Params for xgbTree
xgb.params <- getModelInfo("xgbTree")
xgb.params
xgb.params$xgbTree$parameters

#Params for nnet
nnet.params <- getModelInfo("nnet")
nnet.params
nnet.params$nnet$parameters


#######Model Training

train.control <- trainControl(method = "repeatedcv", number = 10,
                              repeats = 3, search = "grid")

#Creates the permutations for all the values passed 
#For xgb.params$xgbTree$grid in this case
tune.grid <- expand.grid(nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         eta = c(0.05, 0.075, 0.1),
                         gamma = 0,
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         min_child_weight = c(2.0, 2.25, 2.5),
                         subsample = 1
)
head(tune.grid)

#Using 10 k folds and 3 repats times 243 different tune.grid instances

library(doSNOW)
cl <- makeCluster(3, type = "SOCK")

#tells caret that it can now use the available clusters for processing
registerDoSNOW(cl)


caret.cv <- train(Survived ~ ., data = titanic.train, method = "xgbTree",
                  tuneGrid = tune.grid, trControl = train.control)
stopCluster(cl)
caret.cv


#Doing prediction
preds <- predict(caret.cv, titanic.test[,-1])
preds
result <- confusionMatrix(preds, titanic.test$Survived)
result

##Comparing Multiple Caret Models

cl <- makeCluster(3, type="SOCK")
registerDoSNOW(cl)

######  Using random forest without a tuneGrid
caret.rf <- train(Survived ~ .,
                  data = titanic.train,
                  method = "rf",
                  #tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)
pred2 <- predict(caret.rf, titanic.test[,-1])
confusionMatrix(pred2, titanic.test$Survived)

######  Using generalized linear model
cl <- makeCluster(3, type="SOCK")
registerDoSNOW(cl)
caret.nnet <- train(Survived ~ .,
                    data = titanic.train,
                    method = "glm",
                    #tuneGrid = tune.grid,
                    trControl = train.control)
stopCluster(cl)
pred3 <- predict(caret.nnet, titanic.test[,-1])
confusionMatrix(pred3, titanic.test$Survived)





