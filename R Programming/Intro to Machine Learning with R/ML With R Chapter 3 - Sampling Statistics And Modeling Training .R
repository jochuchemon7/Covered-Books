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


########### CHAPTER 3 SAMPLING STATISTICS AND MODELING TRAINING ###############

# Random normal distribution plotting
example <- cbind(data.table(rnorm(n = 100,0,1)), data.table(rnorm(100,0,1)))
plot(example)


## Checking for Vairiance and Bias

data <- data.table(iris)
sample.index <- sample(1:nrow(data), nrow(data) * 0.75, replace = FALSE)
head(data[sample.index, ])
summary(data)
summary(data[sample.index,])


#Systematic Sampling

sys.sample = function(N, n) {
  k = ceiling(N/n)
  r = sample(1:k, 1)
  sys.samp = seq(r, r + k * (n - 1), k)
}
systematic.index <- sys.sample(nrow(iris), nrow(iris) * 0.75)
summary(iris[systematic.index, ])



set.seed(123)
x <- rnorm(100, 2, 1)

y = exp(x) + rnorm(5, 0, 2)
plot(x, y)
linear <- lm(y ~ x)
abline(a = coef(linear[1], b = coef(linear[2], lty = 2)))
summary(linear)


data <- data.table(x, y)
data.samples <- sample(1:nrow(data), nrow(data) * 0.7, replace = FALSE)
training.data <- data[data.samples, ]
test.data <- data[-data.samples, ]

train.linear <- lm(y ~ x, training.data)
train.output <- predict(train.linear, test.data)

#This
RMSE.df = data.frame(predicted = train.output, actual = test.data$y,
                     SE = ((train.output - test.data$y)^2/length(train.output)))
head(RMSE.df)
sqrt(sum(RMSE.df$SE))

#Or This
RMSE(train.output, test.data$y)


#To the second degree
train.quadratic <- lm(y ~ x^2 + x, training.data)
quadratic.output <- predict(train.quadratic, test.data)
RMSE.quad.df = data.frame(predicted = quadratic.output, actual = test.data$y,
                          SE = ((quadratic.output - test.data$y)^2/length(train.output)))
head(RMSE.quad.df)
sqrt(sum(RMSE.quad.df$SE))



#To the fourth degree
train.polyn <- lm(y ~ poly(x, 4), training.data)
polyn.output <- predict(train.polyn, test.data)
RMSE.polyn.df = data.frame(predicted = polyn.output, actual = test.data$y,
                           SE = ((polyn.output - test.data$y)^2/length(train.output)))
head(RMSE.polyn.df)
sqrt(sum(RMSE.polyn.df$SE))

#Different Regression Metrics
RMSE(polyn.output, test.data$y)  #Avg differences
rae(test.data$y, polyn.output)   # 0-1 scale
MAE(polyn.output, test.data$y)   #Avg differences
rrse(test.data$y, polyn.output)  # 0-1 scale


# Confusion Matrix
# Use table() or confusionMatrix()

## Cross Validation - K Fold

set.seed(123)
x <- rnorm(100, 2, 1)
y = exp(x) + rnorm(5, 0, 2)
data <- data.table(x, y)
plot(data)
data.shuffled <- data[sample(nrow(data)), ]
folds <- cut(seq(1, nrow(data)), breaks = 10, labels = FALSE)
errors <- c(0)

for (i in 1:10) {
  fold.indexes <- which(folds == i, arr.ind = TRUE)
  test.data <- data[fold.indexes, ]
  training.data <- data[-fold.indexes, ]
  train.linear <- lm(y ~ x, training.data)
  train.output <- predict(train.linear, test.data)
  errors <- c(errors, RMSE(train.output, test.data$y))
}

errors[2:11]

