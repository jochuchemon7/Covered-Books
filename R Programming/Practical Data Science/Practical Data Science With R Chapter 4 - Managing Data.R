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

 Chapter 4 Managing Data

#Load exampleData.rData gives new  custdata, hhdata & medianincome
load(file = "Desktop/R Work/Learning/Practical Data Science With R/Data/exampleData.rData")

#To get summary of recent.move and num.vehicles where housing.type if NA valued
summary(custdata[is.na(custdata$housing.type), c("recent.move","num.vehicles")])

#Same as above using which
summary(custdata[which(is.na(custdata$housing.type)), c("recent.move", "num.vehicles")])

#Check all columns for any na value
apply(custdata,2, function(x) any(is.na(x)))

#328 entries on is.employed has NA values (about 1/3 of the data)
summary(as.factor(custdata$is.employed))

#Make a new columns that says if employed, not employed of missing (rather than just NA)
custdata$is.employed.fix <- ifelse(is.na(custdata$is.employed), "missing",
                                   ifelse(custdata$is.employed == TRUE, "employed", "not employed"))

#Summary on the new column
summary(as.factor(custdata$is.employed.fix))

#Or from "missing" to -> "not in active workforce"
custdata$is.employed.fix <- ifelse(is.na(custdata$is.employed),"not in active workforce",
                                   ifelse(custdata$is.employed==T,"employed", "not employed"))
summary(as.factor(custdata$is.employed.fix))

#Summary on Income table has 328 NA values
summary(custdata$Income)

#Get the mean income from existing incomes
meanIncome <- mean(custdata$Income, na.rm=T)

#Replace the NA values with the mean 
Income.fix <- ifelse(is.na(custdata$Income), meanIncome, custdata$Income)

#Summary once again
summary(Income.fix)

###When values are Systematically missing

#Create income categories in a sense of ranges
breaks <-c(0, 10000, 50000, 100000, 250000, 1000000)

#Place incomes in to the designateed breaks 
Income.groups <- cut(custdata$Income, breaks=breaks, include.lowest=T)

#Summary
summary(Income.groups)

#From factor to character
Income.groups <- as.character(Income.groups)

#Replace NAs to no income while values are characters
Income.groups <- ifelse(is.na(Income.groups),"no income", Income.groups)

#Summary as factor
summary(as.factor(Income.groups))

#Gets index of na values for Income (for tracking purposes)
missingIncome <- is.na(custdata$Income)

#Replaces all NA to 0 (but we keep index for values that NA to -> 0)
Income.fix <- ifelse(is.na(custdata$Income), 0, custdata$Income)


### Summary from medianincome df
summary(medianincome)

#Merge the medianincome df with custdata (now medianincome is in custdata)
#state.of.res -> State & Median.Income -> Median.Income.y & Median.Income.x
NewCustdata <- merge(x = custdata, y = medianincome, by.x="state.of.res", by.y="State")

#Summary for state.of.res, income, Median.Income for custdata and NewCustdata
summary(custdata[,c("state.of.res", "income", "Median.Income")])
summary(NewCustdata[,c("state.of.res", "income", "Median.Income.y")])

#Create a column with the rate of income/Medican.Income (Normalized)
custdata$income.norm <- with(data = custdata, expr = income/Median.Income)

#Summary of the rate
summary(custdata$income.norm)

#An Example of nonsense
example <- custdata %>%
  select(state.of.res, Median.Income) %>%
  dplyr::group_by(state.of.res) %>%
  summarise(Median.Income = mean(Median.Income))
example <- as.data.frame(example)

####-----Converting continous variables to discrete

#Add column T/F if individual´s income is less than $20,000 dollars
custdata$income.lt.20K <- custdata$income < 20000

#Summary of it (about 1/3 have an income less than $20,000 dollars)
summary(custdata$income.lt.20K)

###Converting ages into ranges
breaks = c(0,25,65,Inf)

#Create a new column with the designated range (include.lowests includes 0)
custdata$age.range <- cut(custdata$age, breaks = breaks, include.lowest = T)

#Summary on the factor on the ranges of the ages
summary(custdata$age.range)


###-------- Normalization and Rescaling

#Summary
summary(custdata$age)

#Mean of ages
meanage <- mean(custdata$age)

#Normalized the data using the mean on each age
custdata$age.normalized <- custdata$age/meanage

#Summary on the normalized age (you can see how far from the average customer age
# the rest of the customers are) Closer to one the less uncommon
summary(custdata$age.normalized)

#Standard deviation of the custdata$age
stdage <- sd(custdata$age)

#Mean
meanage

#Standard deviation
stdage

#Rescale by using the standard deviation as a unit of distance
custdata$age.normalized <- (custdata$age-meanage)/stdage

#Summary of it (less than -1 are younger than typical and 1 older than typical)
summary(custdata$age.normalized)

#Using mean and sd normalizations are usefull when the data is roughly symmetric


###----------LOG transformations for skewed and wide distributions

#signdlog10 takes the negatives value dropped by log10

####------SAMPLING FOR MODELING AND VALIDATION

#Random uniform size of the rows of the custdata
custdata$gp <- runif(dim(custdata)[1])

#Subsett the custdata into the train and test data
testSet <- subset(custdata, custdata$gp <= 0.1)
trainingSet <- subset(custdata, custdata$gp > 0.1)

#### Train/Test data ALTERNATIVE METHOD #1
trainData <- custdata[which(custdata$gp > .1),]
testData <- custdata[which(custdata$gp <= .1),]

### Train/Test data ALTERNATIVE METHOD #2
datacontrol <- sample(x = 1:nrow(custdata), size = .89*nrow(custdata), replace = F)
dataTrain <- custdata[datacontrol,]
dataTest <- custdata[-datacontrol,]

#nrowsfor test and train data sets
dim(testSet)[1]
dim(trainingSet)[1]

#Essentially levels(hhdata$house_id) since it is a factor (unique household_id)
hh <- unique(hhdata$household_id)

#data frame with household_id and random uniform on the lenght of hh
households <- data.frame(household_id = hh, gp = runif(length(hh)))

#Remove so we can place ours from the dataframe using that runif()
hhdata$gp = NULL

#Giving the hhdata the households data merging by household_id (gives gp)
hhdata <- merge(hhdata, households, by="household_id")


####################################### SAVES THE custdata df into my database  ####
library(RMySQL)
driver <- dbDriver("MySQL")
con <- RMySQL::dbConnect(drv = driver,dbname = "Practical_Data_Science_Data",user = "Josue", 
                         host = "localhost",password = "diciembre07", port = 3306)
dbListTables(conn = con)
dbWriteTable(conn = con, name = "custdata", value = custdata)
dbDisconnect(conn = con)
####################################### NO NEED TO RUN THIS  #############



