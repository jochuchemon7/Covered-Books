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
library(MASS)
library(ROCR)
library(grid)
library(ggpubr)

#####################Chapter 7 Linear and Logistic Regression


###-----Using Linear Regression

#Loading PUMS data
setwd("Desktop/R Work/Learning/Practical Data Science With R/Data/")
load("psub.RData")
setwd("/Users/Josue")

#Train and Test Data
dtrain <- subset(psub,ORIGRANDGROUP >= 500)
dtest <- subset(psub,ORIGRANDGROUP < 500)


#Logarithmic Linear Regression Model (You can include factors as variables)
model <- lm(log(PINCP,base=10) ~ AGEP + SEX + COW + SCHL,data=dtrain)

#Predict for train and test data again
dtest$predLogPINCP <- predict(model,newdata=dtest)
dtrain$predLogPINCP <- predict(model,newdata=dtrain)

#Plotting log income as a function of predicted log income
ggplot(data=dtest,aes(x=predLogPINCP,y=log(PINCP,base=10))) +
  geom_point(alpha=0.2,color="black") +
  geom_smooth(aes(x=predLogPINCP, 
                  y=log(PINCP,base=10)),color="black") +
  geom_line(aes(x=log(PINCP,base=10),
                y=log(PINCP,base=10)),color="blue",linetype=2) +
  scale_x_continuous(limits=c(4,5)) +
  scale_y_continuous(limits=c(3.5,5.5))


#Plotting residuals income as a function of predicted log income
ggplot(data=dtest,aes(x=predLogPINCP,
                      y=predLogPINCP-log(PINCP,base=10))) +
  geom_point(alpha=0.2,color="black") +
  geom_smooth(aes(x=predLogPINCP,
                  y=predLogPINCP-log(PINCP,base=10)),
              color="black")

#Computing R Square (For measure of quality)
rsq <- function(y,f) { 1 - sum((y-f)^2)/sum((y-mean(y))^2) }
rsq(log(dtrain$PINCP,base=10),predict(model,newdata=dtrain))
rsq(log(dtest$PINCP,base=10),predict(model,newdata=dtest))

#Calculating root mean square error
rmse <- function(y, f) { sqrt(mean( (y-f)^2 )) }
rmse(log(dtrain$PINCP,base=10),predict(model,newdata=dtrain))
rmse(log(dtest$PINCP,base=10),predict(model,newdata=dtest))

#Give out the coefficients in the outcome function (Can also see the weights on
# each coefficient)
coefficients(model)

#Summary of the model
summary(model)

#Summarizing residuals for train and test data (same as summary(model$residual))
#According to what data was used for fitting the model
summary(log(dtrain$PINCP, 10) - predict(model, dtrain))
summary(log(dtest$PINCP,base=10) - predict(model,newdata=dtest))

#The degrees of freedom
df <- dim(dtrain)[1] - dim(summary(model)$coefficients)[1]
#Or
df <- nrow(dtrain) - length(model$coefficients)
df

#To avoid overfitting the degrees of freedom should be larger than the num of coef

#Calculates the residual error
modelResidualError <- sqrt(sum(residuals(model)^2)/df)
modelResidualError

###----Logistic Regression 

#Loading the CDC Data
setwd("Desktop/R Work/Learning/Practical Data Science With R/Data/")
load("NatalRiskData.rData")
setwd("/Users/Josue")

head(sdata)

#Train and Test Data
train <- sdata[sdata$ORIGRANDGROUP<=5,]
test <- sdata[sdata$ORIGRANDGROUP>5,]

#Building Logistic Regression Model
complications <- c("ULD_MECO","ULD_PRECIP","ULD_BREECH")
riskfactors <- c("URF_DIAB", "URF_CHYPER", "URF_PHYPER",
                 "URF_ECLAM")
y <- "atRisk"
#All colnames to be used
x <- c("PWGT","UPREVIS","CIG_REC","GESTREC3","DPLURAL",complications, riskfactors)

#Making the formula
fmla <- as.formula(paste(y, " ~ ", paste(x, collapse = " + "), sep = ''))

#Training our logistic regression model (logit since atRisk -> class is logical)
model <- glm(fmla, data = train, family = binomial("logit"))
model
summary(model)

#Predict the probability the train and test data (response for probability)
train$pred <- predict(model, newdata=train, type="response")
test$pred <- predict(model, newdata=test, type="response")

#Plot the probability density and color the T/F of the atRisk
ggplot(train, aes(x=pred, color=atRisk, linetype=atRisk)) +
  geom_density()


#Create the prediction object with probability pred and the T/F label
predObj <- prediction(train$pred, train$atRisk)

#Create the performance on the prediction object with prec & rec measure
precObj <- performance(predObj, measure="prec") #Precision
recObj <- performance(predObj, measure="rec") #Recall 

#For S4 objects use the @ to extract attributes. Get the 
precision <- (precObj@y.values)[[1]]
prec.x <- (precObj@x.values)[[1]] #x.values are the same for prec & rec Objects
recall <- (recObj@y.values)[[1]]

#Data frame with the precision, prec.x and recall
rocFrame <- data.frame(threshold=prec.x, precision=precision, recall=recall)

#Function to plot multiple plots on a page
nplot <- function(plist) {
  n <- length(plist)
  grid.newpage()
  pushViewport(viewport(layout=grid.layout(n,1)))
  vplayout=function(x,y) {viewport(layout.pos.row=x, layout.pos.col=y)}
  for(i in 1:n) {
    print(plist[[i]], vp=vplayout(i,1))
  }
}

#The percentage of TRUE entries in the atRisk column for all the entries
pnull <- mean(as.numeric(train$atRisk)) #Rate of at risk births in the train data 
#Same as above
as.numeric(summary(train$atRisk)[[3]]) / nrow(train)

#Plot enrichment rate as a function of threshold.
p1 <- ggplot(rocFrame, aes(x=threshold)) +
  geom_line(aes(y=precision/pnull)) +
  coord_cartesian(xlim = c(0,0.05), ylim=c(0,10)) #With probabilities xlim()

#Second Plot
p2 <- ggplot(rocFrame, aes(x=threshold)) +
  geom_line(aes(y=recall)) +
  coord_cartesian(xlim = c(0,0.05))

#Use function to plot the two ggplot graphs in a single page
nplot(list(p1, p2))

#Using ggpubr (without the need of the nplot() function)
ggpubr::ggarrange(p1,p2, ncol = 1, nrow = 2)

###------Evaluating our choosen model

#Table with probabilities bigger than .02 and the T/F of the atRisk column for 
# the test data #Here we give our threshold
ctab.test <- table(pred=test$pred>0.02, atRisk=test$atRisk)
ctab.test

#Precision
precision <- ctab.test[2,2]/sum(ctab.test[2,])
precision

#Recall
recall <- ctab.test[2,2]/sum(ctab.test[,2])
recall

#Enrich
enrich <- precision/mean(as.numeric(test$atRisk))
enrich

#Or have the table make into a confusion matrix when not using factors 
result <- confusionMatrix(ctab.test)
result


#The model coefficients
coefficients(model)
#Any remainder level ei (SEXM *not coefficient) then that is the reference level
# To interpreter the coefficients use exp() since the probs was log(,10)

#Model Summary
summary(model)

#Calculating Deviance Residual
pred <- predict(model, newdata=train, type="response")

llcomponents <- function(y, py) {
  y*log(py) + (1-y)*log(1-py)
}


edev <- sign(as.numeric(train$atRisk) - pred) *
  sqrt(-2*llcomponents(as.numeric(train$atRisk), pred))

summary(edev)

###Computing deviance

#loglikelihood function
loglikelihood <- function(y, py) {
  sum(y * log(py) + (1-y)*log(1 - py))
}

#Calculate rate of positive examples in dataset. 
pnull <- mean(as.numeric(train$atRisk))
#Same as above
table(train$atRisk)[[2]] / nrow(train)

#Calculate Null Deviance
null.dev <- -2*loglikelihood(as.numeric(train$atRisk), pnull)
pnull
null.dev

#Check you have the same number for the null.dev
model$null.deviance

#Predict on the train data
pred <- predict(model, newdata=train, type="response")

#Calculate the residual deviance
resid.dev <- -2*loglikelihood(as.numeric(train$atRisk), pred)
resid.dev

model$deviance

#Change the T/F to numeric (1/0)
testy <- as.numeric(test$atRisk)

#Predict on the test data set now
testpred <- predict(model, newdata=test, type="response")
testpred

#
pnull.test <- mean(testy)
#Same as above
table(testy)[[2]] / length(testy)

#Calculate the loglikelihood on the pnull.test and testpred
null.dev.test <- -2*loglikelihood(testy, pnull.test)
resid.dev.test <- -2*loglikelihood(testy, testpred)

#Show them
pnull.test
null.dev.test
resid.dev.test

#Calculating the significance of the observed fit
df.null <- dim(train)[[1]] - 1
df.model <- dim(train)[[1]] - length(model$coefficients)

df.null
df.model

delDev <- null.dev - resid.dev
deldf <- df.null - df.model

p <- pchisq(delDev, deldf, lower.tail=F)
delDev
deldf
p

#Psuedo R sqare residual
pr2 <- 1-(resid.dev/null.dev)
print(pr2)
pr2.test <- 1-(resid.dev.test/null.dev.test)
print(pr2.test)


aic <- 2*(length(model$coefficients) - 
            loglikelihood(as.numeric(train$atRisk), pred))
#The lower the AIC value the better the model
AIC(model)

#Bottom you can check the number of fisher scoring iterations: #, usually converges
# after 6 to 8 iterations
summary(model)


####--Take aways

#Logistic regression for binary classification 
#Correlated variables lowers the quality of advise
