library(caret)
library(dplyr)
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

#####################Chapter 5 Choosing and Evaluating Models

###-----Evaluation models

#NULL -> for lower bound (what low performance looks like)
#Bayes Rate -> for upper bound (what high performance looks like)

###-----Evaluating Classification Models (Using Logistic Regression)


spamD <- read.table(file = "Desktop/R Work/Learning/Practical Data Science With R/Data/spamD.tsv",
                    sep = "\t", header = T)
head(spamD)
summary(spamD)

#Gets the train based on rgroup being more than or equal to 10
spamTrain <- subset(spamD,spamD$rgroup>=10)
example <- spamD[which(spamD$rgroup >= 10),] #Same as spamTrain above

#For the test sample on rgroup less than 10
spamTest <- subset(spamD,spamD$rgroup<10)
example <- spamD[which(spamD$rgroup < 10),] #Same as spamTest above

#Gives colnames from spamD that do not match with rgroup and spamD
spamVars <- setdiff(colnames(spamD),list('rgroup','spam'))

#Writting the formula that includes all colnames/variables (except rgroup & spam)
spamFormula <- as.formula(paste('spam == "spam"', paste(spamVars, collapse = " + "),
                                sep = " ~ ")) #Given "spam is a Spam" and not just "spam ~"

#Passing the formula to glm and building the model
spamModel <- glm(formula = spamFormula, family = binomial("logit"), data = spamTrain)

#Prediction on the train and test data (having type = "response" for prediction on 
# the probability it is a spam)
spamTrain$pred <- predict(spamModel, newdata = spamTrain, type = "response")
spamTest$pred <- predict(spamModel, newdata = spamTest, type = "response")

#Have any with a probaility more than .5 to be a spam 
spamTest$predSpam <- as.factor(ifelse(spamTest$pred > .5, "spam", "non-spam"))

#Makes the confusion matrix
result <- confusionMatrix(spamTest$spam, spamTest$predSpam)
result #You can see the accuracy, sensetivity, recall and more (including the table)

#***Books method for the table***
with(spamTest,table(y=spam,glmPred=pred>0.5))

#Sample of the classifier 
sample <- spamTest[c(7,35,224,327),c('spam','pred')]
print(sample)

#The book's creation of their confusion matrix 
cM <- table(truth = spamTest$spam, prediction = spamTest$pred > 0.5)
cM

#Their accuracy measure method
(cM[1,1]+cM[2,2])/sum(cM) 

#Their precision measure method -> Specificity on ours (confusionMatrix)
cM[2,2]/(cM[2,2]+cM[1,2]) #Predicted "Spam" correclty (positive comes out correct)
###*******MORE SERIOUS SINCE IMPORTANT DATA CAN BE REMOVED**********

#Their recall measure method -> Neg Pred Value on ours
cM[2,2]/(cM[2,2]+cM[2,1]) #Predicted "non-spam" correctly (negative comes out correct)

#F1 Score = 2*precision*recall/(precision+recall)

###------Example of plottinng residuals for Evaluating Scoring Models

#Data
d <- data.frame(y=(1:10)^2,x=1:10)

#Linear Model
model <- lm(y~.,data=d)

#Predict on new data
d$prediction <- predict(model,newdata=d)
d$prediction <- as.integer(d$prediction)

#Plot the line (predicted model) acrosss the data points
ggplot(data=d) + 
  geom_point(aes(x=x,y=y)) +
  geom_line(aes(x=x,y=prediction),color='blue') +
  geom_segment(aes(x=x,y=prediction,yend=y,xend=x)) + #Connects line to points
  scale_y_continuous('')

#RMSE (Root Mean Square Error)
RMSE(pred = d$prediction, obs = d$y)

#Book's RMSE method
sqrt(mean((d$prediction-d$y)^2))

#R^2
summary(model)

#Books Method for R^2
1-sum((d$prediction-d$y)^2)/sum((mean(d$y)-d$y)^2)

###----Evaluating Probability Models

#To turn off the scientific notation
format(spamTest$pred, scientific = F)
##*****Or to disable scientific*******
options(scipen=999)
###*******To return to scientific********
options(scipen = 0)

#Plot a density graph on the pred (probability) pred probability with density of spam
# And no spam (Spam tends to have a pred probability of .75 > )
ggplot(spamTest)+
  geom_density(aes(x = pred, color = spam, linetype = spam))


####-------ROC Curve

library('ROCR')
eval <- ROCR::prediction(predictions = spamTest$pred, labels = spamTest$spam)

#Plot the "balance" between sensitivity and specification
plot(ROCR::performance(eval,"tpr","fpr"))
print(attributes(performance(eval,'auc'))$y.values[[1]])

###--- Log Likelihood (BOOKS CODE)

#Adds all the log(probability). Model log likelihood the model assigns to the 
#test data. Always a negative, the clsoer to 0 the better
sum(ifelse(spamTest$spam=='spam', log(spamTest$pred), log(1-spamTest$pred)))


#log likelihood rescale to the number of data points, gives avg surprise per data point
sum(ifelse(spamTest$spam=='spam',log(spamTest$pred), 
           log(1-spamTest$pred)))/dim(spamTest)[[1]]

##Copmuting the null model's log method (A good Null model)
pNull <- sum(ifelse(spamTest$spam=='spam',1,0))/dim(spamTest)[[1]]

#Gives log likehood on the NULL method = 306.89 so our log likelihood is better
# than the NULL method (just guessing)
sum(ifelse(spamTest$spam=='spam',1,0))*log(pNull) +
  sum(ifelse(spamTest$spam=='spam',0,1))*log(1-pNull)





####----- Log Likelihood Method (MY CODE)
#Measure loglikelihood
loglikelihood <- ifelse(spamTest$spam == "spam", log(spamTest$pred), log(1-spamTest$pred))
sum(loglikelihood)

#Loglikelihood per data point (entry)
sum(loglikelihood)/nrow(spamTest)
datapointLog <- ifelse(spamTest$spam == "spam", log(spamTest$pred), 
                       log(1 -spamTest$pred))/nrow(spamTest)
sum(datapointLog) #total surprice per data point 

#NULL log likelihood method
nullValue <- sum(ifelse(spamTest$spam == "spam", 1, 0))/nrow(spamTest)
nullValue

#NULL likelihood (just guessing 50/50)
nullLikelihood <- sum(ifelse(spamTest$spam == "spam", 1,0))*log(nullValue)+
  sum(ifelse(spamTest$spam == "spam", 0,1))*log(1-nullValue)
nullLikelihood


####---------Calculating Entropy and Conditional Entropy------####

#Entropy function
entropy <- function(x) {
  xpos <- x[x>0]
  scaled <- xpos/sum(xpos) #Each value spam & non spam, divided by nrows
  sum(-scaled*log(scaled,2))
}

#Print entropy function
print(entropy(table(spamTest$spam)))

#Conditional Entropy Function
conditionalEntropy <- function(t) {
  (sum(t[,1])*entropy(t[,1]) + sum(t[,2])*entropy(t[,2]))/sum(t)
}

#Print conditional entropy function
print(conditionalEntropy(cM))

###---------Evaluating Clustering Models (Using kmeans)------###

####### Use of the do.call function (Similar to apply)
f <- function(x) print(x^2)
A <- c(2,3,4,10,1912,3231,2)
do.call(what = f, args = list(A)) #Args must be a list
########

#Set seed
set.seed(32297)
#Data
data <- data.frame(x = runif(100), y = runif(100))
#Model
fit <- stats::kmeans(x = data, centers = 4)
fit
#Add the assigned cluster to each point
data$cluster <- fit$cluster
#Libraries
library('ggplot2')
library('grDevices')

######****NOT NECESSATRY BUT OKAY FOR ENCAPSULATING CLUSTERS GRAPHICALLY*******
#Each unique cluster number, used to subset data with those values on their cluster
fence <- lapply(unique(data$cluster), #Then calculate chull points (perimeter of cluster)
                function(x){clusSet <- data[which(data$cluster == x),];
                clusSet <- clusSet[chull(clusSet),]})
fence <- do.call(rbind, fence) #Applies rbind to the fence list(do.call -> for lists)
fence #holds the fence points group by cluster value
#####################***********

#Witouth convex hull
ggplot(data = data)+
  geom_point(aes(x =x, y=y, colour = cluster))

#With convex hull
ggplot(data = data)+
  geom_text(aes(x=x, y=y, label = cluster, color = cluster), size =3)+
  geom_polygon(data = fence, aes(x=x,y=y,group=cluster, fill = as.factor(cluster)),
               alpha = .4, linetype = 0)+
  theme(legend.position = "none")

#Table on clusters
table(data$cluster)

#######################---BOOK'S METHOD ON KMEANS---################################
#Data
d <- data.frame(x=runif(100),y=runif(100))
#kmean method for clustering
clus <- kmeans(d,centers=5)
#Save the values of cluter assign to the point 
d$cluster <- clus$cluster

#Creating the convex hull of the clusters
h <- do.call(rbind, lapply(unique(clus$cluster), 
                           function(c) { f <- subset(d,cluster==c); f[chull(f),]}))
#Plot the clusterring
ggplot() +
  geom_text(data=d,aes(label=cluster,x=x,y=y, color=cluster),size=3) +
  geom_polygon(data=h,aes(x=x,y=y,group=cluster,
                          fill=as.factor(cluster)),alpha=0.4,linetype=0) +
  theme(legend.position = "none")
#table on the number of clusters
table(d$cluster)
###################################################################################

###Calculating the typical distance between items in every pair of clusters
library('reshape2')

#nrows(d)
n <- dim(d)[[1]]

#Distance of each point to every other point
pairs <- data.frame(ca = as.vector(outer(1:n,1:n,function(a,b) d[a,'cluster'])),
                    cb = as.vector(outer(1:n,1:n,function(a,b) d[b,'cluster'])),
                    dist = as.vector(outer(1:n,1:n,function(a,b)
                      sqrt((d[a,'x']-d[b,'x'])^2 + (d[a,'y']-d[b,'y'])^2))))

#The mean distance from points in one cluster to points in another. 
#Diagonal values are the intra cluster mean distance (smaller than the intra)
# Ei. other clusters
dcast(pairs,ca~cb,value.var='dist',mean)


####-----Model Validation

#We have bayesian inferance and frequentist inference 

#Significance test have err.null and err.model as long as err.null - err.model > 0
#The model has done better than the null model (average, guessing)

#For classifiers a good significance test is the fisher.test()

#frequency inference -> confidence of interval (% that the predicted val is intended)
#bayesian inference -> credible intervals




