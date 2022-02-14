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



#################### Chapter 6 Tree-Based Methods #####################

library(caret)
library(randomForest)

mtcars.rf <- randomForest(mpg ~ ., data = mtcars, ntree = 1000,
                          keep.forest = FALSE, importance = FALSE)

varImpPlot(mtcars.rf)
plot(mtcars.rf, log = "y", title = "")


### Pruning Our Tree

library(rpart)

head(cu.summary)
data <- cu.summary

#You want a regression model for Mileage
# Recursive Parttitioning and Regression Tree
fit <- rpart(
  Mileage~Price + Country + Reliability + Type,
  method="anova", #method="class" for classificaiton tree
  data=data
)
plot(fit, uniform=TRUE, margin=0.1)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


rsq.rpart(fit)[1]
plotcp(fit) 

fit$cptable

#Pruned tree using xerror as the complexity parameter for the pruning
fit.pruned <- prune(fit, 
                    cp = fit$cptable[which.min(fit$cptable[,"xerror"]), "CP"])
par(mfrow = c(1, 2))

plot(fit, uniform = TRUE, margin = 0.1, main = "Original Tree")
text(fit, use.n = TRUE, all = TRUE, cex = 0.8)

plot(fit.pruned, uniform = TRUE, margin = 0.1, main = "Pruned Tree")
text(fit.pruned, use.n = TRUE, all = TRUE, cex = 0.8)



##########  Desicion Tree for Regression

#complete.cases for cases with no missing values
cu.summary.complete <- cu.summary[complete.cases(cu.summary),]

data.samples <- sample(1:nrow(cu.summary.complete), nrow(cu.summary.complete) *
                         0.7, replace = FALSE)

training.data <- cu.summary.complete[data.samples, ]
test.data <- cu.summary.complete[-data.samples, ]

fit <- rpart(
  Mileage~Price + Country + Reliability + Type,
  method="anova", #method="class" for classification tree
  data=training.data
)

plotcp(fit) 
plot(fit, uniform=TRUE, margin=0.1)
text(fit, use.n=TRUE, all=TRUE, cex=.8)
fit$cptable


fit.pruned<- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

par(mfrow = c(1, 2))
plot(fit, uniform = TRUE, margin = 0.1, main = "Original Tree")
text(fit, use.n = TRUE, all = TRUE, cex = 0.8)

plot(fit.pruned, uniform = TRUE, margin = 0.1, main = "Pruned Tree")
text(fit.pruned, use.n = TRUE, all = TRUE, cex = 0.8)


prediction <- predict(fit.pruned, test.data[,-4])
output <- data.frame(test.data$Mileage, prediction)
output
RMSE(prediction, test.data$Mileage)


##########  Desicion Tree for Classification

cu.summary.complete <- cu.summary[complete.cases(cu.summary),]

data.samples <- sample(1:nrow(cu.summary.complete), nrow(cu.summary.complete) *
                         0.7, replace = FALSE)

training.data <- cu.summary.complete[data.samples, ]
test.data <- cu.summary.complete[-data.samples, ]

fit <- rpart(Type ~ Price + Country + Reliability + Mileage,
             method = "class", #Class for CLassification
             data = training.data)

plotcp(fit) 

fit.pruned <- prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]), "CP"])

par(mfrow = c(1, 2))
plot(fit, uniform = TRUE, margin = 0.1, main = "Original Tree")
text(fit, use.n = TRUE, all = TRUE, cex = 0.8)

plot(fit.pruned, uniform = TRUE, margin = 0.1, main = "Pruned Tree")
text(fit.pruned, use.n = TRUE, all = TRUE, cex = 0.8)

prediction <- predict(fit.pruned, test.data, type = "class")
prediction
result <- confusionMatrix(prediction, test.data$Type)
result


#### Conditional Inference Tree

library(party)
fit2 <- ctree(Mileage ~ Price + Country + Reliability + Type,
              data = na.omit(cu.summary))
plot(fit2) #For Regression


fit3 <- ctree(Type ~ Price + Country + Reliability + Mileage,
              data = na.omit(cu.summary))
plot(fit3) #For Classification


#### Conditional Inference Tree Regression

set.seed(123)
cu.summary.complete <- cu.summary[complete.cases(cu.summary),]

data.samples <- sample(1:nrow(cu.summary.complete), nrow(cu.summary.complete) *
                         0.7, replace = FALSE)
training.data <- cu.summary.complete[data.samples, ]
test.data <- cu.summary.complete[-data.samples, ]

fit.ctree <- ctree(Mileage ~ Price + Country + Reliability + Type,
                   data = training.data)
plot(fit.ctree)

prediction.ctree <- predict(fit.ctree, test.data)
RMSE(prediction.ctree, test.data$Mileage)

#### Conditional Inference Tree Classification

set.seed(456)
data.samples <- sample(1:nrow(cu.summary), nrow(cu.summary) *
                         0.7, replace = FALSE)

training.data <- cu.summary[data.samples, ]
test.data <- cu.summary[-data.samples, ]

fit.ctree <- ctree(Type ~ Price + Country + Reliability + Mileage,
                   data = training.data)
prediction.ctree <- predict(fit.ctree, test.data)
result <- confusionMatrix(test.data$Type, prediction.ctree)
result


### Random Forest Regression

library(randomForest)
set.seed(123)
cu.summary.complete <- cu.summary[complete.cases(cu.summary),]
data.samples <- sample(1:nrow(cu.summary.complete), nrow(cu.summary.complete) *
                         0.7, replace = FALSE)

training.data <- cu.summary.complete[data.samples, ]
test.data <- cu.summary.complete[-data.samples, ]

fit.rf <- randomForest(Mileage ~ Price + Country + Reliability +
                         Type, data = training.data)
plot(fit.rf)
prediction.rf <- predict(fit.rf, test.data)
prediction.rf
RMSE(prediction.rf, test.data$Mileage)


#### Random Forest For Classification

cu.summary.complete <- cu.summary[complete.cases(cu.summary),]
samples <- sample(1:nrow(cu.summary.complete), nrow(cu.summary.complete)*.7,
                  replace = FALSE)

training.data <- cu.summary.complete[samples,]
test.data <- cu.summary.complete[-samples,]

fit.rf <- randomForest(Type ~ ., data = training.data)
fit.rf
plot(fit.rf)

pred <- predict(fit.rf, newdata = test.data[,-5])
pred 
result <- confusionMatrix(pred, test.data$Type)
result$overall

