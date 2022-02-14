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


################ CHAPTER 4 REGRESSION IN A NUTSHELL #############################

#Linear Reg

model <- lm(mtcars$mpg ~ mtcars$disp)
plot(y = mtcars$mpg, x = mtcars$disp, xlab = "Engine Size (cubic inches)",
     ylab = "Fuel Efficiency (Miles per Gallon)", main = "Fuel Efficiency From
     the `mtcars` Dataset")
abline(a = coef(model[1]), b = coef(model)[2], lty = 2)
summary(model)

## Multivariate Regression

lm.wt <- lm(mpg ~ disp + wt, data = mtcars)
summary(lm.wt)

lm.cyl <- lm(mpg ~ disp + wt + cyl, data = mtcars)
summary(lm.cyl)

lm.cyl.wt <- lm(mpg ~ wt + cyl, data = mtcars)
summary(lm.cyl.wt)


## Regularization

#Lasso
library(lasso2)
lm.lasso <- l1ce(mpg ~ ., data = mtcars)
summary(lm.lasso)$coefficients

summary(lm.lasso)

lm.lasso2 <- l1ce(mpg ~ cyl + hp + wt + am + carb, data = mtcars)
summary(lm.lasso2)$coefficients

lm.lasso3 <- l1ce(mpg ~ cyl + hp + wt, data = mtcars)
summary(lm.lasso3)$coefficients

lm.lasso4 <- l1ce(mpg ~ cyl + wt, data = mtcars)
summary(lm.lasso4)$coefficients

## Eliminate the coefficients that are close to 0 for their value

## Polynomial Regression

pop <- data.table(uspop)
pop$uspop <- as.numeric(pop$uspop)
pop$year <- seq(from = 1790, to = 1970, by = 10)
plot(y = pop$uspop, x = pop$year, main = "United States Population From 1790 to
     1970", xlab = "Year", ylab = "Population")

lm1 <- lm(pop$uspop ~ pop$year)
summary(lm1)
abline(a = coef(lm1[1]), b = coef(lm1)[2], lty = 2, col = "red")

## Higher degree model

lm2 <- lm(pop$uspop ~ poly(pop$year, 2))
summary(lm2)

plot(y = pop$uspop, x = pop$year, main = "United States Population From 1790 to
     1970", xlab = "Year", ylab = "Population")
pop$lm2.predict = predict(lm2, newdata = pop)
lines(sort(pop$year), fitted(lm2)[order(pop$year)], col = "blue", lty = 2)

lm3 <- lm(pop$uspop ~ poly(pop$year,3))
lm4 <- lm(pop$uspop ~ poly(pop$year,4))
lm5 <- lm(pop$uspop ~ poly(pop$year,5))
lm6 <- lm(pop$uspop ~ poly(pop$year,6))

par(mfrow = c(2, 3))
plot(resid(lm1), main = "Degree 1", xlab = "Sequential Year", ylab = "Fit Residual")
plot(resid(lm2), main = "Degree 2", xlab = "Sequential Year", ylab = "Fit Residual")
plot(resid(lm3), main = "Degree 3", xlab = "Sequential Year", ylab = "Fit Residual")
plot(resid(lm4), main = "Degree 4", xlab = "Sequential Year", ylab = "Fit Residual")
plot(resid(lm5), main = "Degree 5", xlab = "Sequential Year", ylab = "Fit Residual")
plot(resid(lm6), main = "Degree 6", xlab = "Sequential Year", ylab = "Fit Residual")

# To Check 
c(sum(abs(resid(lm1))), sum(abs(resid(lm2))), sum(abs(resid(lm3))),
  sum(abs(resid(lm4))), sum(abs(resid(lm5))), sum(abs(resid(lm6))))



# We can also check the RMSE for all the models



# Goodness Of Fit

table((summary(lm1)$coefficients[, 4]) < 0.05)


model.order <- c(1,2,3,4,5,6)
coef.true <- c(
  table((summary(lm1)$coefficients[,4])<0.05) - 1
  ,table((summary(lm2)$coefficients[,4])<0.05) - 1
  ,table((summary(lm3)$coefficients[,4])<0.05)[2] - 1
  ,table((summary(lm4)$coefficients[,4])<0.05)[2] - 1
  ,table((summary(lm5)$coefficients[,4])<0.05)[2] - 1
  ,table((summary(lm6)$coefficients[,4])<0.05)[2] - 1
)
coef.false <- c(
  0
  ,0
  ,table((summary(lm3)$coefficients[,4])<0.05)[1]
  ,table((summary(lm4)$coefficients[,4])<0.05)[1]
  ,table((summary(lm5)$coefficients[,4])<0.05)[1]
  ,table((summary(lm6)$coefficients[,4])<0.05)[1]
)
model.rsq <- c(
  summary(lm1)$r.squared
  ,summary(lm2)$r.squared
  ,summary(lm3)$r.squared
  ,summary(lm4)$r.squared
  ,summary(lm5)$r.squared
  ,summary(lm6)$r.squared
)
model.comparison <- data.frame(model.order, model.rsq, coef.true, coef.false)
model.comparison$goodness <- (model.comparison$coef.true / model.comparison
                              $model.order)
model.comparison

# Motivation for Classification

data <- data.frame(tumor.size <- c(1, 2, 3, 4, 5, 6, 7, 8, 9,20), 
                   malignant <- c(0, 0, 0, 0, 1, 1, 1, 1, 1, 1))
tumor.lm <- lm(malignant ~ tumor.size, data = data)
plot(y = data$malignant, x = data$tumor.size, main = "Tumor Malignancy by Size",
     ylab = "Type (0 = benign, 1 = cancerous)", xlab = "Tumor Size")
abline(a = coef(tumor.lm[1]), b = coef(tumor.lm[2]))
#Or
abline(a = tumor.lm$coefficients[1], b = tumor.lm$coefficients[2])
coef(tumor.lm)

summary(tumor.lm)$r.squared

#The decision boundary

plot(y = data$malignant, x = data$tumor.size, main = "Tumor Malignancy by Size",
     ylab = "Type (0 = benign, 1 = cancerous)", xlab = "Tumor Size")
abline(v = 4.5)

#Sigmoid Funct
e <- exp(1)
curve(1/(1 + e^-x), -10, 10, main = "The Sigmoid Function", xlab = "Input",
      ylab = "Probability")

lengths <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
t1 = -4.5
t2 = 1
g = t1 + t2 * lengths
s = 1/(1 + e^-g)
data.frame(lengths, g, s)

plot(y = s, x = lengths, pch = 1, main = "Sigmoid Function Inputs and Rounding
     Estimates", xlab = "Tumor Lengths", ylab = "Probability of Class 1 Typification")
points(y = round(s), x = lengths, pch = 3)



# Binary Classification

plot(iris$Sepal.Length ~ iris$Sepal.Width, main = "Iris Flower Sepal Length vs
     Sepal Width", xlab = "Sepal Width", ylab = "Sepal Length")

iris.binary <- iris
iris.binary$binary <- as.numeric(iris[, 5] == "setosa") #Only is Setosa
iris.logistic <- glm(binary ~ Sepal.Width + Sepal.Length, data = iris.binary,
                     family = "binomial")
iris.logistic

#Calculate slope and intercept from logistic reg
slope.iris <- coef(iris.logistic)[2]/(-coef(iris.logistic)[3])
int.iris <- coef(iris.logistic)[1]/(-coef(iris.logistic)[3])
iris.binary$binary[iris.binary$binary == 0] <- 2
plot(Sepal.Length ~ Sepal.Width, data = iris.binary, pch = (binary),
     main = "Iris Flower Sepal Length vs Sepal Width", xlab = "Sepal Width",
     ylab = "Sepal Length")
abline(a = int.iris, b = slope.iris)


iris.binary$binary[iris.binary$binary == 0] <- 2
plot(Sepal.Length ~ Sepal.Width, data = iris.binary, pch = (binary),
     main = "Iris Flower Sepal Length vs Sepal Width", xlab = "Sepal Width",
     ylab = "Sepal Length")
abline(a = int.iris, b = slope.iris)

## Multiclass Classification

multi <- data.frame(x1 = c(0.03, 0.24, 0.21, 0, 0, 0.23, 0.6,0.64, 0.86, 0.77), 
                    x2 = c(0.07, 0.06, 0.19, 1.15, 0.95, 1, 0.81, 0.64, 0.44, 0.74),
                    lab = c(1, 1, 1, 2, 2, 2, 3, 3,3, 3))
plot(x2 ~ x1, pch = lab, cex = 2, data = multi, main = "Multi-Class Classification",
     xlab = "x", ylab = "y")

par(mfrow = c(1, 3)) # To have multiple graphs in a single image
multi$lab2 <- c(1, 1, 1, 4, 4, 4, 4, 4, 4, 4)
plot(x2 ~ x1, pch = lab2, cex = 2, data = multi, main = "Multi-Class Classification",
     xlab = "x", ylab = "y")

library(nnet)
multi.model <- multinom(lab ~ x2 + x1, data = multi, trace = F)
multi.model
summary(multi.model)


#Logistic Regression With Caret

library(caret)
data("GermanCredit")
Train <- createDataPartition(GermanCredit$Class, p = 0.6, list = FALSE)
training <- GermanCredit[Train, ]
testing <- GermanCredit[-Train, ]
mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate +
                   Housing.Own + CreditHistory.Critical, data = training,
                 method = "glm",
                 family = "binomial")
predictions <- predict(mod_fit, testing[, -10])
table(predictions, testing[, 10])
matrixs <- confusionMatrix(predictions, testing[,10])
matrixs



mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate +
                   Housing.Own + CreditHistory.Critical, data = training,
                 method = "LogitBoost",
                 family = "binomial")
predictions <- predict(mod_fit, testing[, -10])
table(predictions, testing[, 10])

