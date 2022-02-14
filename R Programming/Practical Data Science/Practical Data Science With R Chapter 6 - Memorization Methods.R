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

#####################Chapter 6 Memorization Methods

#The simpliest methods ^,  generate answers by returning a majority category for 
#classification or average value for scoring ei. knn, naive bayes, descicion trees

#Original wd -> /Users/Josue
getwd() #Set for data read purposes
setwd("Desktop/R Work/Learning/Practical Data Science With R/Data/")

#Total KDD data
d <- read.table('orange_small_train.data.gz', header=T, sep='\t',
                na.strings=c('NA','')) #Treat NA & '' as missing data

#Rea the churn -> cancelled account, appetency ->tendency to use new products and
# upselling -> willingness to respond favorative to market pitches. All Data labels 
churn <- read.table('orange_small_train_churn.labels.txt', header=F,sep='\t')
appetency <- read.table('orange_small_train_appetency.labels.txt', header=F,sep='\t')
upselling <- read.table('orange_small_train_upselling.labels.txt', header=F,sep='\t')

#Return to original wd -> "/Users/Josue"
setwd("/Users/Josue/")

#Create a column for each data read label an add it to d (totla KDD data)
d$churn <- churn$V1
d$appetency <- appetency$V1
d$upselling <- upselling$V1

set.seed(729375) #Makes our work reproducible so any else gets the same result

#setting the random index random uniform numbers
d$rgroup <- runif(dim(d)[[1]])

#Set the train and test data
dTrainAll <- d[which(d$rgroup <= .9),]
dTest <- d[which(d$rgroup > .9),]

###############-----DIFFERENT WAY OF GETTING TRAIN AND TEST DATA-----################
#Gets index on random points in 80% of the data
trainIndex <- sample(x = 1:nrow(d), size = .8*(nrow(d)))
#All index values are unique to use
length(unique(trainIndex))
#Train and Test data
trainData <- d[trainIndex,]
testData <- d[-trainIndex,]
#####################################################################################

#List of the table names that where added where we have an output
outcomes=c('churn','appetency','upselling')

#Variables that can be used we removed the outcome columns (predicted results cols)
#Along side with rgroup the 'primary key' for each entry
vars <- setdiff(colnames(dTrainAll), c(outcomes,'rgroup'))

#We have the colnames whose data is classified as character or factor
catVars <- vars[sapply(dTrainAll[vars], class) %in% c("character", "factor")]
#Or
alternativeCatVars <- vars[sapply(dTrainAll, is.factor)]

#Print categorical values
catVars

#Get columns name that have numerica values in the column
numericVars <- vars[sapply(dTrainAll[vars], class) %in% c("integer", "numeric")]

#Remove the d, churn, appetency and upselling data
rm(list=c('d','churn','appetency','upselling'))

#Select Which outcome to model to 
outcome <- 'churn'

#Which outcome is consider possitive (Binomial outcome in this case)
pos <- '1'

#Calibration Index
useForCal <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)>0
#Calibration data subset 
dCal <- subset(dTrainAll,useForCal)
#Train data subset
dTrain <- subset(dTrainAll,!useForCal)

############################--- SAME AS ABOVE --- #################################
#Split training data into training and calibration data
calibrationControl <- sample(x = 1:nrow(dTrainAll), size = .9*(nrow(dTrainAll)))
#Train and Calibration data
trainData <- dTrainAll[calibrationControl, ]
calibrationData <- dTrainAll[-calibrationControl,]
###################################################################################


####----Building single-variable models

#Table on with the dtrain$var218, current desire outcome columns
table218 <- table(Var218=dTrain[,'Var218'],
  churn=dTrain[,outcome],
  useNA='ifany') #To include NA values if any

#Also known as the contingency table
#Table of the binary outcome on churn counting the levels on var218 (NA also)
table218

#Percentange of occurences for each level of Var218 when churn is 1 (Is Cancelled)
print(table218[,2]/(table218[,1]+table218[,2]))
#E.i 6% of people who had cJvF churned (cancelled)


#Function to predict outcome based on MUMBO JUMBO on Categorical variables
mkPredC <- function(outCol,varCol,appCol) {#Pred result,train varibles,new data var 
  pPos <- sum(outCol==pos)/length(outCol) #Rate of churn on dTrain
  naTab <- table(as.factor(outCol[is.na(varCol)]))#table of outcome churn for na values 
  pPosWna <- (naTab/sum(naTab))[pos]#Percentage of positive outcome on the na values
  vTab <- table(as.factor(outCol),varCol)#Churn distribution for each level on that variable
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}

#Applies out pred function to all categorical variables and 
#Spliting the data prediction into the train, calibration and test data sets
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dCal[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dCal[,v])
  dTest[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTest[,v])
}

#Example of the pred results on the train data
pred <- gsub(pattern = "Var", replacement = "predVar", x = catVars)
head(dTrain[, pred])

library('ROCR')

calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

#When the auc for the train is more than .8 it tests performance on the calibration
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

ggplot(dTrain, aes(x = predVar192))+
  geom_histogram()


#Make Pred For numeric Variables
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

#Make the prediction on the numerica variables and shwo auc for those whoe train
# the auc is more than .55 
for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dTest[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTest[,v])
  dCal[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dCal[,v])
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

#Plot variable performance
ggplot(data=dCal) +
  geom_density(aes(x=predVar126,color=as.factor(churn)))

#Running a repeated cross-validation experiment
var <- 'Var217'

#For 100 times of cross validation
aucs <- rep(0,100)

#Calculate 
for(rep in 1:length(aucs)) { #Selects some 10% of data on each time for test data
  useForCalRep <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)>0
  predRep <- mkPredC(dTrainAll[!useForCalRep,outcome],
                     dTrainAll[!useForCalRep,var],
                     dTrainAll[useForCalRep,var])
  aucs[rep] <- calcAUC(predRep,dTrainAll[useForCalRep,outcome])
}

#Mean if the auc for multiple cross validation (100) with 10% of test data
#has a very close auc for the original auc is pretty good 
mean(aucs)

#Standard deviation 
sd(aucs)

#Or Use a function so it can be easier to replicate 
fCross <- function() {
  useForCalRep <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)>0
  predRep <- mkPredC(dTrainAll[!useForCalRep,outcome],
                     dTrainAll[!useForCalRep,var],
                     dTrainAll[useForCalRep,var])
  calcAUC(predRep,dTrainAll[useForCalRep,outcome])
}

#same result but using the replicate function rather than for loop
aucs <- replicate(100,fCross())



####-------Bulding Models Using Multiple Variables

#Log likelyhood function
logLikelyhood <-function(outCol,predCol) {
    sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

#Variable names
selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(dCal[,outcome],
                               sum(dCal[,outcome]==pos)/length(dCal[,outcome]))

#If 2 * (loglikelihood - base rate check) is more than the minStep all on the 
# calibration data and on the categorical variables only
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}

#Same as above but for the numeric variables and also for the calibration data
for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck) - 1)
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


###----Building a Descision Tree

#Recursive partitioning and regression tree
library('rpart')

#Formula "churn>0 ~ " all categorical and numerica variables
fV <- paste(outcome,'>0 ~ ', paste(c(catVars,numericVars),collapse=' + '),sep='')

#Train the recursive partitioning and regression tree with the dtrain data
tmodel <- rpart(fV,data=dTrain)

#Predict the train, calibration and test data using the rpart model on the dTrain
print(calcAUC(predict(tmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(tmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(tmodel,newdata=dCal),dCal[,outcome]))
#Really bad AUC results


#Gathers all the PredVar# that where categorical and numeric 
tVars <- paste('pred',c(catVars,numericVars),sep='')
fV2 <- paste(outcome, '>0 ~ ', paste(tVars, collapse= ' + '), sep= '')

#Trains the new decision tree
tmodel <- rpart(fV2,data=dTrain)

#Print the auc of predicted val and real outcome for train, cali and test data
print(calcAUC(predict(tmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(tmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(tmodel,newdata=dCal),dCal[,outcome]))
#Very similar auc for test and valibration data while very good on the train 
#Data that has seen in the past bc of not every level from train is on test/cali data

#Another tree model but with control parameter; complexity = .001, minsplit and
# minbucket = 1000, and maxdepth = 5
tmodel <- rpart(fV2,data=dTrain,control=rpart.control(cp=0.001,minsplit=1000,
                                      minbucket=1000,maxdepth=5))

#Prints similar results
print(calcAUC(predict(tmodel, newdata = dTrain), dTrain[,outcome]))
print(calcAUC(predict(tmodel, newdata = dTest), dTest[,outcome]))
print(calcAUC(predict(tmodel, newdata = dCal), dCal[,outcome]))

#Amother tree but with predict values whose predict had the best likelihood score
f <- paste(outcome,'>0 ~ ',paste(selVars,collapse=' + '),sep='')
tmodel <- rpart(f,data=dTrain, control=rpart.control(cp=0.001,minsplit=1000,
                                        minbucket=1000,maxdepth=5))

print(calcAUC(predict(tmodel, newdata = dTrain), dTrain[,outcome]))
print(calcAUC(predict(tmodel, newdata = dTest), dTest[,outcome]))
print(calcAUC(predict(tmodel, newdata = dCal), dCal[,outcome]))
#As good as we can get it 

tmodel

#Plot the tree model
par(cex=0.7)
plot(tmodel)
text(tmodel)

##Using nearest neighbor methods
library('class')

nK <- 200

#Using the pred variables and values as the knnTrain. Only variables we wish for the
#classifier
knnTrain <- dTrain[,selVars]

#and the -1/1 of churn outcome
knnCl <- dTrain[,outcome]==pos

#Knn predict function
knnPred <- function(df) {
  #knn() - trainData with desired attributes only, the newdata,true classifiers on
  #the trainData, number of neighboors consider, and to return the proportion of votes
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T) #for the winning class
  ifelse(knnDecision==TRUE,
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}

#Results
print(calcAUC(knnPred(dTrain[,selVars]),dTrain[,outcome]))
print(calcAUC(knnPred(dCal[,selVars]),dCal[,outcome]))
print(calcAUC(knnPred(dTest[,selVars]),dTest[,outcome]))

#Gets the knnPred value 
dCal$kpred <- knnPred(dCal[,selVars])

#Plotting 200-nearest neighbor performance
ggplot(data=dCal) +
  geom_density(aes(x=kpred,color=as.factor(churn),linetype=as.factor(churn)))

#Plotting the receiver operating characteristic curve
plotROC <- function(predcol,outcol) {
  
  perf <- performance(prediction(predcol,outcol==pos),'tpr','fpr')
  pf <- data.frame(FalsePositiveRate=perf@x.values[[1]], 
                   TruePositiveRate=perf@y.values[[1]])
  
  ggplot() +
    geom_line(data=pf,aes(x=FalsePositiveRate,y=TruePositiveRate)) +
    geom_line(aes(x=c(0,1),y=c(0,1)))
}
print(plotROC(knnPred(dTest[,selVars]),dTest[,outcome]))

#log regression on the probability variables 
gmodel <- glm(as.formula(f),data=dTrain,family=binomial(link='logit'))

print(calcAUC(predict(gmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(gmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(gmodel,newdata=dCal),dCal[,outcome]))

###Building, applying, and evaluating a Naive Bayes model

#Positive 
pPos <- sum(dTrain[,outcome]==pos)/length(dTrain[,outcome])

nBayes <- function(pPos,pf) {
  pNeg <- 1 - pPos
  smoothingEpsilon <- 1.0e-5
  scorePos <- log(pPos + smoothingEpsilon) +
    rowSums(log(pf/pPos + smoothingEpsilon))
  scoreNeg <- log(pNeg + smoothingEpsilon) +
    rowSums(log((1-pf)/(1-pPos) + smoothingEpsilon))
  m <- pmax(scorePos,scoreNeg)
  expScorePos <- exp(scorePos-m)
  expScoreNeg <- exp(scoreNeg-m)
  expScorePos/(expScorePos+expScoreNeg)
}

pVars <- paste('pred',c(numericVars,catVars),sep='')

dTrain$nbpredl <- nBayes(pPos,dTrain[,pVars])
dCal$nbpredl <- nBayes(pPos,dCal[,pVars])
dTest$nbpredl <- nBayes(pPos,dTest[,pVars])

print(calcAUC(dTrain$nbpredl,dTrain[,outcome]))
print(calcAUC(dCal$nbpredl,dCal[,outcome]))
print(calcAUC(dTest$nbpredl,dTest[,outcome]))


#Using the naive bayes library
library('e1071')

lVars <- c(catVars,numericVars)

#The formula containing the 'as.factor(churn>0) ~ lVars'
ff <- paste('as.factor(',outcome,'>0) ~ ', paste(lVars,collapse=' + '),sep='')

#Train the naive bayes model
nbmodel <- naiveBayes(as.formula(ff),data=dTrain)

#Predict using the model, and the new data having type = 'raw'
dTrain$nbpred <- predict(nbmodel,newdata=dTrain,type='raw')[,'TRUE']
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
dTest$nbpred <- predict(nbmodel,newdata=dTest,type='raw')[,'TRUE']

#Calculates the auc
calcAUC(dTrain$nbpred,dTrain[,outcome])
calcAUC(dCal$nbpred,dCal[,outcome])
calcAUC(dTest$nbpred,dTest[,outcome])



