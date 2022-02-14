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



#################### Chapter 7 Other Advance Methods #####################


##### Naive Bayes Classifier

data <- fread("Data/breast-cancer-wisconsin.data")
data
namesData <- c("Sample code number", 
               "Clump Thickness" ,
               "Uniformity of Cell Size" ,
               "Uniformity of Cell Shape",      
               "Marginal Adhesion",
               "Single Epithelial Cell Size",   
               "Bare Nuclei" ,                  
               "Bland Chromatin",               
               "Normal Nucleoli" ,              
               "Mitoses" ,                        
               "Class"                       
)

colnames(data) <- namesData
colnames(data)
head(data)

data <- data.frame(sapply(data, as.factor))
breast_cancer_features <- data[, 2:11]

breast_cancer_complete <-
  breast_cancer_features[complete.cases(breast_cancer_features),]

breast_cancer_complete$Class <- as.factor(breast_cancer_complete$Class)

data.samples <- sample(1:nrow(breast_cancer_complete),
                       nrow(breast_cancer_complete) *
                         0.7, replace = FALSE)

training.data <- breast_cancer_complete[data.samples, ]
test.data <- breast_cancer_complete[-data.samples, ]

nb.model <- naiveBayes(Class ~ ., data = training.data)
nb.model
summary(nb.model)

prediction.nb <- predict(nb.model, test.data)
prediction.nb
results <- confusionMatrix(prediction.nb, test.data$Class)
results


####### Principal Component Analysis (PCA)

head(mtcars)

pairs(mtcars[, 1:7], lower.panel = NULL)

pca <- princomp(mtcars, scores = TRUE, cor = TRUE)
pca

#Interested in the proportion of variance (How much of the data is explain by
# that component)
summary(pca)
plot(pca)

#To see how much of each variable is used for 1 to 5 compenent
pca$loadings[, 1:5]



scores.df <- data.frame(pca$scores)
scores.df$car <- row.names(scores.df)

#Plotting the scores of the components
plot(x = scores.df$Comp.1, y = scores.df$Comp.2, xlab = "Comp1 (mpg,cyl)",
     ylab = "Comp2 (qsec, gear, am)")

text(scores.df$Comp.1, scores.df$Comp.2, labels = scores.df$car,
     cex = 0.7, pos = 3)

#Certain cars fall in a certain spectrum and other in another spectrum



############# Linear Discriminant Analysis (LDA)

iris.pca <- prcomp(iris[, -5], center = T, scale. = T)

#The total variance attributed to each of the components  
iris.pca$sdev^2/sum(iris.pca$sdev^2)
#Component 1 and 2 are the main important ones

#3 Classes equally distributed
table(iris$Species)

library(MASS)

# Assing the Distributions for Species
iris.lda <- lda(Species ~ ., data = iris, prior = c(1/3, 1/3, 1/3))
iris.lda$svd^2/sum(iris.lda$svd^2)

#How each Linear Discriminat is related to each variable 
iris.lda$scaling


iris.lda.prediction <- predict(iris.lda, newdata = iris)
result <- confusionMatrix(iris.lda.prediction$class, iris$Species)
result



#### LDA and PCA Differences
#PCA - Unsuprevised
#LDA - Supervised

combined <- data.frame(Species = iris[, "Species"], pca = iris.pca$x,
                       lda = iris.lda.prediction$x)
combined

library(ggplot2)
library(gridExtra)


lda.plot <- ggplot(combined) + geom_point(aes(lda.LD1, lda.LD2, shape = Species))
+ scale_shape_manual(values = c(0, 1, 2))

pca.plot <- ggplot(combined) + geom_point(aes(pca.PC1, pca.PC2, shape = Species))
+ scale_shape_manual(values = c(0, 1, 2))

grid.arrange(pca.plot, lda.plot)

#LDA Separates all 3 better than PCA does


############ Support Vector Machines

library("e1071")
s <- sample(150, 100)
col <- c("Petal.Length", "Petal.Width", "Species")

iris_train <- iris[s, col]
iris_test <- iris[-s, col]

svmfit <- svm(Species ~ ., data = iris_train, kernel = "linear",
              cost = 0.1, scale = FALSE)
svmfit
plot(svmfit, iris_train)


#Tune function to help find the best cost parameter for optimal tuning
tuned <- tune(svm, Species ~ ., data = iris_train, kernel = "linear",
              ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)))
summary(tuned)
#Best cost is .1



###Curve Descision for classification

str(cats)
plot(x = cats$Hwt, y = cats$Bwt, pch = as.numeric(cats$Sex))


library(MASS)
library(e1071)

data(cats)
model <- svm(Sex ~ ., data = cats)
print(model)
summary(model)

plot(model, cats)

#For the Standard Confusion Matrix

data.samples <- sample(1:nrow(cats), nrow(cats) * 0.7, replace = FALSE)

training.data <- cats[data.samples, ]
test.data <- cats[-data.samples, ]

svm.cats <- svm(Sex ~ ., data = training.data)
plot(svm.cats, training.data)

prediction.svm <- predict(svm.cats, test.data[, -1], type = "class")
prediction.svm

result <- confusionMatrix(prediction.svm, test.data[,1])
result


###### K-Nearest Neighboors


knn.ex <- head(mtcars[, 1:3])
knn.ex

knn.ex$dist <- sqrt((knn.ex$cyl - 6)^2 + (knn.ex$disp - 225)^2)
knn.ex[order(knn.ex[, 4]), ]


## Regression with KNN

library(caret)
data(BloodBrain)

inTrain <- createDataPartition(logBBB, p = 0.8)[[1]]

trainX <- bbbDescr[inTrain, ]
trainY <- logBBB[inTrain]

testX <- bbbDescr[-inTrain, ]
testY <- logBBB[-inTrain]

fit <- knnreg(trainX, trainY, k = 3)
summary(fit)

result  <- predict(fit, testX)
plot(testY, result)
RMSE(result, testY)


## Classification with KNN

library(RWeka)
iris <- read.arff(system.file("arff", "iris.arff", package = "RWeka"))

classifier <- IBk(class ~ ., data = iris)
summary(classifier)

classifier <- IBk(class ~ ., data = iris, control = Weka_control(K = 20,
                                                                 X = TRUE))
evaluate_Weka_classifier(classifier, numFolds = 10)



