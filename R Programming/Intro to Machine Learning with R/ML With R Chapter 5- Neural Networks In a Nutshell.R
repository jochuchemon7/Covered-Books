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

################ Chapter 5 Neural Networks In A Nutshell ########################

x1 <- c(0, 0, 1, 1)
x2 <- c(0, 1, 0, 1)
logic <- data.frame(x1, x2)
logic$AND <- as.numeric(x1 & x2)
logic

#### NeuralNet Library

library(neuralnet)
set.seed(123)
AND <- c(rep(0, 3), 1)
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1)), AND)
net <- neuralnet(AND ~ Var1 + Var2, binary.data, hidden = 0,
                 err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")
prediction(net)



set.seed(123)
AND <- c(rep(0, 7), 1)
OR <- c(0, rep(1, 7))
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1), c(0,1)), AND, OR)
net <- neuralnet(AND + OR ~ Var1 + Var2 + Var3, binary.data,
                 hidden = 0, err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")
prediction(net)                  



### Hidden layers 

set.seed(123)
AND <- c(rep(0, 7), 1)
binary.data <- data.frame(expand.grid(c(0, 1), c(0, 1), c(0,1)), AND, OR)
net <- neuralnet(AND ~ Var1 + Var2 + Var3, binary.data, hidden = 1,
                 err.fct = "ce", linear.output = FALSE)
plot(net, rep = "best")
prediction(net)


net2 <- neuralnet(AND ~ Var1 + Var2 + Var3, binary.data, hidden = 2,
                  err.fct = "ce", linear.output = FALSE)
plot(net2, rep = "best")
prediction(net)

## With 4 and 8 hiddden layers
net4 <- neuralnet(AND ~ Var1 + Var2 + Var3, binary.data, hidden = 4,
                  err.fct = "ce", linear.output = FALSE)
net8 <- neuralnet(AND ~ Var1 + Var2 + Var3, binary.data, hidden = 8,
                  err.fct = "ce", linear.output = FALSE)
plot(net4, rep = "best")
plot(net8, rep = "best")

#With two outputs and hidden 
net6 <- neuralnet(AND + OR ~ Var1 + Var2 + Var3, binary.data,
                  hidden = 6, err.fct = "ce", linear.output = FALSE)
plot(net6, rep = "best")
prediction(net6)

#Multilayer NN
x1 <- c(0, 0, 1, 1)
x2 <- c(0, 1, 0, 1)
logic <- data.frame(x1, x2)
logic$AND <- as.numeric(x1 & x2) + 1
logic$OR <- as.numeric(x1 | x2) + 1
logic
par(mfrow = c(2, 1))
plot(x = logic$x1, y = logic$x2, pch = logic$AND, cex = 2,
     main = "Simple Classification of Two Types",
     xlab = "x", ylab = "y", xlim = c(-0.5, 1.5), ylim = c(-0.5,1.5))

plot(x = logic$x1, y = logic$x2, pch = logic$OR, cex = 2,
     main = "Simple Classification of Two Types",
     xlab = "x", ylab = "y", xlim = c(-0.5, 1.5), ylim = c(-0.5,1.5))



###NN For Regression

# Standard LM
library(mlbench)
data(BostonHousing)
lm.fit <- lm(medv ~ ., data = BostonHousing)
lm.predict <- predict(lm.fit)
plot(BostonHousing$medv, lm.predict, main = "Linear regression predictions vs actual",
     xlab = "Actual", ylab = "Prediction")

#Using NN

library(nnet)
nnet.fit1 <- nnet(medv ~ ., data = BostonHousing, size = 2)
nnet.predict1 <- predict(nnet.fit1)
plot(BostonHousing$medv, nnet.predict1, main="Neural network predictions vs actual",
     xlab = "Actual", ylab = "Prediction")

summary(BostonHousing$medv)

##### Edit
samples <- BostonHousing$medv %>%
  createDataPartition(p =.8, list = FALSE)

train <- BostonHousing[samples,-4]
test <- BostonHousing[-samples,-4]

summary(train$medv)
summary(train$medv/50)

train$medv <- train$medv/50
test$medv <- test$medv/50

model <- neuralnet(medv ~ ., data = train, hidden = 2, err.fct = "sse", 
                   linear.output = TRUE)
model
summary(model)
pred <- predict(model, newdata = test[,-13]) 
pred
RMSE(pred, test$medv)
######


#We need to do feature scaling to normalize the data from the range of 5-50

summary(BostonHousing$medv/50) #We divide by 50 in this case 


nnet.fit2 <- nnet(medv/50 ~ ., data = BostonHousing, size = 2,
                  maxit = 1000, trace = FALSE)
nnet.predict2 <- predict(nnet.fit2) * 50
plot(BostonHousing$medv, nnet.predict2, main = "Neural network predictions vs
     actual with normalized response inputs",
     xlab = "Actual", ylab = "Prediction")

RMSE(pred = nnet.predict2, obs = BostonHousing$medv)

mean((lm.predict - BostonHousing$medv)^2)
mean((nnet.predict2 - BostonHousing$medv)^2)


library(caret)
mygrid <- expand.grid(.decay = c(0.5, 0.1), .size = c(4, 5, 6))
nnetfit <- train(medv/50 ~ ., data = BostonHousing, method = "nnet",
                 maxit = 1000, tuneGrid = mygrid, trace = F)
print(nnetfit)

#### NN For Classification

iris.df <- iris
smp_size <- floor(0.75 * nrow(iris.df))
set.seed(123)
train_ind <- sample(seq_len(nrow(iris.df)), size = smp_size)
train <- iris.df[train_ind, ]
test <- iris.df[-train_ind, ]

#Two types of methods

#Num 1
iris.nnet <- nnet(Species ~ ., data = train, size = 4, decay = 0.0001,
                  maxit = 500, trace = FALSE)
predictions <- predict(iris.nnet, test[, 1:4], type = "class")
table(predictions, test$Species)



### NN For Regression Using Caret

library(car)
library(caret)
trainIndex <- createDataPartition(Prestige$income, p = 0.7, list = F)
prestige.train <- Prestige[trainIndex, ]
prestige.test <- Prestige[-trainIndex, ]
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6,7))

prestige.fit <- train(income ~ prestige + education, data = prestige.train,
                      method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F,
                      linout = 1)
prestige.predict <- predict(prestige.fit, newdata = prestige.test)

summary(prestige.test$income)
sqrt(mean((prestige.predict - prestige.test$income)^2))
RMSE(prestige.predict, prestige.test$income)


#Using another NN Method

neuralnetFit <- train(income ~ prestige + education, data = prestige.train,
                      method = "neuralnet") 
neuralnetFit
neuralnetPred <- predict(neuralnetFit, newdata = prestige.test[,-2])
neuralnetPred

sqrt(mean((neuralnetPred - prestige.test$income)^2))
RMSE(neuralnetPred, prestige.test$income)


newNeuralNetFit <- neuralnet(income ~ prestige + education, data = prestige.train, 
                             err.fct = "sse", hidden = 3, linear.output = TRUE)
newNeuralNetFit
newPred <- predict(newNeuralNetFit, newdata = prestige.test[,-2])
newPred


### NN For Classification With Caret

#Also for regression neuralnet, brnn, qrnn, and mlpSGD
iris.caret <- train(Species ~ ., data = train, method = "nnet", trace = FALSE)
predictions <- predict(iris.caret, test[, 1:4])
predictions
table(predictions, test$Species)

#multinom is for classification specific
iris.caret.m <- train(Species ~ ., data = train, method = "multinom", trace = FALSE)
predictions.m <- predict(iris.caret.m, test[, 1:4])
table(predictions.m, test$Species)

