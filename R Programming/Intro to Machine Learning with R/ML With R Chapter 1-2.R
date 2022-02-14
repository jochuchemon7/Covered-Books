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



barplot(mtcars$mpg, names.arg = row.names(mtcars), las = 3, ylab = "Fuel
        Efficiency in Miles per Gallon")

data <- mtcars
head(data)
row.names(data)

pairs(data[1:7], lower.panel = NULL)
pairs(data[1:7])

DataComputing::scatterGraphHelper(data)
ggplot(data=data[1:7],aes(x=wt,y=mpg))+geom_point() + ylab("Miles Per Gallon") +
  xlab("Weigth in tons") 

fit <- lm(mpg ~ wt, data = data[1:7])  
summary(fit)



####  Logistic ####

plot(x = mtcars$mpg, y = mtcars$am, xlab = "Fuel Efficiency (Miles per Gallon)",
     ylab = "Vehicle Transmission Type (0 = Automatic, 1 = Manual)")

fit2 <- glm(am ~ mpg , data = data)
summary(fit2)


library(caTools)

train.control <- data$am %>%
  createDataPartition(p = .8, list  =FALSE)

train <- data[train.control,]
test <- data[-train.control,]

train.control <- trainControl(method = "cv")
fit4 <- train(as.factor(am) ~ mpg, data= train , method = "glm", 
              trControl = train.control)

summary(fit4)
pred <- predict(fit4, newdata = test[,-9])
pred
matr <- confusionMatrix(pred, as.factor(test$am))

fit3 = LogitBoost(xlearn = train[,-9], ylearn = train[,9])
summary(fit3)
Data.test = test
pred1 <- predict(fit3, test)
matr1 <- confusionMatrix(as.factor(pred1), as.factor(test$am))


##### KMeans

plot(x = iris$Petal.Length, y = iris$Petal.Width, xlab = "Petal Length",
     ylab = "Petal Width")

data1 <- iris

DataComputing::scatterGraphHelper(data1)
ggplot(data=data1,aes(x=Petal.Length,y=Petal.Width))+geom_point() 


irisData = data.frame(data1$Petal.Length, data1$Petal.Width)
colnames(irisData) <- c("Petal Length", "Petal Width")
irisData

### For 2 Clusters
iris.kmeans <- kmeans(irisData, 2)
plot(x = data1$Petal.Length, y = data1$Petal.Width, pch = iris.kmeans$cluster,
     xlab = "Petal Length", ylab = "Petal Width")
points(iris.kmeans$centers, pch = 8, cex = 2)


### For 3 Clusters
iris.kmeans3 <- kmeans(irisData, 3)
plot(x = iris$Petal.Length, y = data1$Petal.Width, pch = iris.kmeans3$cluster,
     xlab = "Petal Length", ylab = "Petal Width")
points(iris.kmeans3$centers, pch = 8, cex = 2)



#### Comparing the actual vs kmeans clustering 
par(mfrow = c(1, 2))

plot(x = data1$Petal.Length, y = data1$Petal.Width, pch = iris.kmeans3$cluster,
     xlab = "Petal Length", ylab = "Petal Width", main = "Model Output")

plot(x = data1$Petal.Length, y = data1$Petal.Width, pch = as.integer(data1$Species),
     xlab = "Petal Length", ylab = "Petal Width", main = "Actual Data")

table(iris.kmeans3$cluster, data1$Species)
kMeansConfusion <- confusionMatrix(as.factor(iris.kmeans3$cluster), as.factor(as.integer(data1$Species)))
kMeansConfusion$table


### Mixed Methods -- Tree Based Clusters

library(party)

data <- mtcars

train.control <- data$am %>%
  createDataPartition(p=.8, list = FALSE)

train <- data[train.control,]
test <- data[-train.control,]

tree <- ctree(mpg ~ ., data = mtcars)
plot(tree)

tree.train <- ctree(mpg ~ ., data = train)
plot(tree.train)

test$mpg.tree <- predict(tree.train, test)
test$class <- predict(tree.train, test, type = "node")
data.frame(row.names(test), test$mpg, test$mpg.tree, test$class)


##### Random Forests

library(randomForest)
mtcars.rf <- randomForest(mpg ~ ., data = mtcars, ntree = 1000,
                          keep.forest = FALSE, importance = FALSE)

plot(mtcars.rf, log = "y", title = "")



###### NN  ####

set.seed(123)
library(nnet)
iris.nn <- nnet(Species ~ ., data = iris, size = 2)
iris.nn
summary(iris.nn)

pred <- predict(iris.nn, iris, type = "class")

table(iris$Species, predict(iris.nn, iris, type = "class"))


#### SVM  #####

library(e1071)
iris.svm <- svm(Species ~ ., data = iris)
table(iris$Species, predict(iris.svm, iris, type = "class"))



data <- iris
train.control <- data$Species %>%
  createDataPartition(p =.8, list =FALSE)
train <- data[train.control,]
test <- data[-train.control,]

correlation <- cor(train[,-5])
correlation
highlyCorrelated <- findCorrelation(correlation, cutoff = .5)
highlyCorrelated
newTrain <- train[,c(3,4,5)]

fit <- svm(Species ~ ., data = newTrain)
fit
pred <- predict(fit, newdata = test[,-5])
length(pred)
class(pred)
confusionMatrix(pred, test$Species)



#### UnSupervised  

x <- rbind(matrix(rnorm(100, sd = 0.3), ncol = 2), matrix(rnorm(100,
                                                                mean = 1, sd = 0.3), ncol = 2))
colnames(x) <- c("x", "y")
plot(x)

cl <- kmeans(x, 2)
plot(x, pch = cl$cluster)

cl
cl$centers






