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

set.seed(500)
library(MASS)
data <- Boston

#NA
apply(data,2,function(x) sum(is.na(x)))

index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

lm.fit <- glm(medv~., data=train)

summary(lm.fit)
pr.lm <- predict(lm.fit,test, response = "")
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)
MSE.lm
RMSE(pr.lm, test$medv)


#########   Setting the NN For Linear Regression #######
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

#Normalized
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

#Train
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
nnpred <- prediction(nn)

#Computes
pr.nn <- compute(nn,test_[,1:13])
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
pr.result <- as.numeric(pr.nn_)
RMSE(pr.result, test$medv)

MSE.nn <- sum((test$medv - pr.nn_)^2)/nrow(test)
MSE.nn

#Plotting
par(mfrow=c(1,2))

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)



##############  Log NeuralNet #############

#Num 1
#### Nos working :(
library(neuralnet)
data <- datasets::airquality
bank = fread("Desktop/R Work/Learning/Samples/Data/bank.csv")

nonCharacter <- function(data){
  
info <- sapply(data, class)
result <- c(1:ncol(data))  
for (i  in 1:ncol(data)) {
  result[i] <- info[i] == "character"
  print(info[i] == "character")
}
result <- as.data.table(result)
result$index <- 1:ncol(data)
final <- c()
for (i in 1:nrow(result)) {
  if(result[i,1] == 1){
    print(result[i,2])
    final <- c(final,result[i,2])
  }  
}
final <- as.numeric(final)
data <- data[,c(-2  ,-3  ,-4  ,-5  ,-7 , -8,  -9, -11, -16)]
return(data)

}

data <- nonCharacter(bank)
data
rm(bank)

data$y <- as.factor(data$y)
data <- as.data.frame(data)
apply(data,2,function(x) sum(is.na(x)))
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

fit <- neuralnet(y ~ ., train, hidden = c(4,2), linear.output = F)
fit <- neuralnet(y ~ ., train, hidden = c(8,4), linear.output = F, rep = 1)
                 #stepmax = 1e+6)
plot(fit)

computed <- compute(fit, test[,-8])
result <- computed$net.result
result

result <- as.data.table("result" <- ifelse(max.col(result[,1:3]) ==1, "yes",
                                     ifelse(max.col(result[,1:3]) ==0, "no")))
#######


#Num2
data <- datasets::ChickWeight
data <- as.data.table(data)
head(data,4)
str(data)

temp <- data
temp$Diet <- as.numeric(temp$Diet)
temp$Chick <- as.numeric(temp$Chick)
correlation <- cor(temp[,c(1,2,3,4)])
correlation

data$Chick <- as.numeric(data$Chick)

samples <- data$Diet %>%
  createDataPartition(p =.8, list = F)

train <- data[samples,]
test <- data[-samples,]
ss <- sample(nrow(test))
test <- test[ss,]
test

fit <- neuralnet(Diet ~ ., train, hidden = c(4,2), linear.output = F)
plot(fit)

computed <- neuralnet::compute(fit, test[,-4])
results <- computed$net.result
results

#They are the same
train$Diet
as.numeric(train$Diet)
round(results)

results = data.frame("results"=ifelse(max.col(results[,1:4])==1, "1",
                                      ifelse(max.col(results[,1:4]) == 2, "2",
                                             ifelse(max.col(results[,1:4]) == 3, "3", "4"))))
results
confusion <- confusionMatrix(results$results, test$Diet)
confusion


#Num3
newData <- iris

datasample <- sample(1:nrow(newData), round(.7*nrow(newData)))
newdatasample <- sample(seq_len(nrow(newData)), size = floor(0.80 * nrow(newData)))

newTrain <- newData[datasample,]
newTest <- newData[-datasample,]
ss <- sample(nrow(newTest))
newTest <- newTest[ss,]
newTest

# IF not converges you can stepmax = 1e+06 and increase rep = 
nnet <- neuralnet(Species~., newTrain, hidden = c(4,3), linear.output = FALSE)
plot(nnet) 

ypred = compute(nnet, newTest[,-5]) 
yhat = ypred$net.result
print(yhat)

result = data.frame("yhat"=ifelse(max.col(yhat[ ,1:3])==1, "setosa",
                              ifelse(max.col(yhat[ ,1:3])==2, "versicolor", "virginica")))

cm=confusionMatrix(as.factor(newTest[,5]), as.factor(result$yhat))
print(cm) 






########## Traditional Logistic ###########
data = fread("Desktop/R Work/Learning/Samples/Data/diabetes.csv")

#Method 1
samples <- sample(seq_len(nrow(data)), size = floor(0.80 * nrow(data)))
train <- data[samples,]
test <- data[-samples,]

fit <- train(as.factor(Outcome) ~ ., data = train, method = "glm")
fit 
pred <- predict(fit, newdata = test[,-9])
pred
result <- confusionMatrix(pred, as.factor(test$Outcome))
result$overall

#Method 2
fit2 <- glm(as.factor(Outcome) ~., data = train, family = binomial(link = "logit")) 
fit2
pred2 <- predict(fit2, newdata = test[,-9], type = "response")
pred2 <- ifelse(pred2 > 0.5,1,0)
pred2
result2 <- confusionMatrix(as.factor(pred2), as.factor(test$Outcome))
result2$overall




