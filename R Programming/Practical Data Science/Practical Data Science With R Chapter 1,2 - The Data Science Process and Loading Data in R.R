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


#####################Chapter 1-2 The Data Science Process and Loading Data in R


#Building the descision tree
library('rpart')
load("Desktop/R Work/Learning/Practical Data Science With R/Data/GCDData.RData")

#rpart -> recursive partitioning and regression trees
model <- rpart(Good.Loan ~
                 Duration.in.month +
                 Installment.rate.in.percentage.of.disposable.income +
                 Credit.amount +
                 Other.installment.plans,
               data=d,
               control=rpart.control(maxdepth=4),
               method="class")


result <- confusionMatrix(as.factor(resultframe$Good.Loan), as.factor(resultframe$pred))
result

#Assuming more than 15% of income is borrowed they default 
tab1
sum(diag(tab1))/sum(tab1)

tab2
sum(diag(tab2))/sum(tab2)


####Chapter 2

#Website give 404 error
uciCar <- read.table('http://www.win-vector.com/dfiles/car.data.csv',sep=',',
                     header=T)

#German credit data 
d <- read.table(paste('http://archive.ics.uci.edu/ml/',
                      'machine-learning-databases/statlog/german/german.data',sep=''),
                stringsAsFactors=F,header=F)
str(d)

##View top three rows
d[1:3,]

colnames(d) <- c('Status.of.existing.checking.account',
                 'Duration.in.month', 'Credit.history', 'Purpose',
                 'Credit.amount', 'Savings account/bonds',
                 'Present.employment.since', 'Installment.rate.in.percentage.of.disposable.income',
                 'Personal.status.and.sex', 'Other.debtors/guarantors',
                 'Present.residence.since', 'Property', 'Age.in.years',
                 'Other.installment.plans', 'Housing',
                 'Number.of.existing.credits.at.this.bank', 'Job',
                 'Number.of.people.being.liable.to.provide.maintenance.for',
                 'Telephone', 'foreign.worker', 'Good.Loan')
d$Good.Loan <- as.factor(ifelse(d$Good.Loan==1,'GoodLoan','BadLoan'))
print(d[1:3,])

#To view help on the class of d
help(class(d))

#Makes a list where you can assing a value to a key and access it later on
mapping <- list('A11'='... < 0 DM',
                'A12'='0 <= ... < 200 DM',
                'A13'='... >= 200 DM / salary assignments for at least 1 year',
                'A14'='no checking account',
                'A30'='no credits taken/all credits paid back duly',
                'A31'='all credits at this bank paid back duly',
                'A32'='existing credits paid back duly till now',
                'A33'='delay in paying off in the past',
                'A34'='critical account/other credits existing (not at this bank)',
                'A40'='car (new)',
                'A41'='car (used)',
                'A42'='furniture/equipment',
                'A43'='radio/television',
                'A44'='domestic appliances',
                'A45'='repairs',
                'A46'='education',
                'A47'='(vacation - does not exist?)',
                'A48'='retraining',
                'A49'='business',
                'A410'='others',
                'A61'='... < 100 DM',
                'A62'='100 <= ... < 500 DM',
                'A63'='500 <= ... < 1000 DM',
                'A64'='.. >= 1000 DM',
                'A65'='unknown/ no savings account',
                'A71'='unemployed',
                'A72'='... < 1 year',
                'A73'='1 <= ... < 4 years',
                'A74'='4 <= ... < 7 years',
                'A75'='.. >= 7 years',
                'A91'='male : divorced/separated',
                'A92'='female : divorced/separated/married',
                'A93'='male : single',
                'A94'='male : married/widowed',
                'A95'='female : single',
                'A101'='none',
                'A102'='co-applicant',
                'A103'='guarantor',
                'A121'='real estate',
                'A122'='if not A121 : building society savings agreement/life insurance',
                'A123'='if not A121/A122 : car or other, not in attribute 6',
                'A124'='unknown / no property',
                'A141'='bank',
                'A142'='stores',
                'A143'='none',
                'A151'='rent',
                'A152'='own',
                'A153'='for free',
                'A171'='unemployed/ unskilled - non-resident',
                'A172'='unskilled - resident',
                'A173'='skilled employee / official',
                'A174'='management/ self-employed/highly qualified employee/ officer',
                'A191'='none',
                'A192'='yes, registered under the customers name',
                'A201'='yes',
                'A202'='no')

#Changes the codes (in charater) and replaces them with the value of the list
#Also it gives then the $ value as the key of the previous list (creates a list)
example <- mapping[d$Credit.history]
#Recomended to change back to character 


for(i in 1:(dim(d))[2]) {
  if(class(d[,i])=='character') {
    d[,i] <- as.factor(as.character(mapping[d[,i]]))
  }
}



for (i  in seq_along(d)) {
  if(class(d[,i]) == "character"){
    d[,i] <- as.factor(as.character(mapping[d[,i]])) 
  }
}

#You can view from the credit history the number of good/bad loans based on the code
table(d$Credit.history, d$Good.Loan)


######## Example of documentation 

# 3-12-2013
# PUMS Data set from:
#   http://www.census.gov/acs/www/data_documentation/pums_data/   <- data source
#   select "2011 ACS 1-year PUMS"     <- How we navigated to get the files
# `select "2011 ACS 1-year Public Use Microdata Samples\
# (PUMS) - CSV format"
# download "United States Population Records" and
# "United States Housing Unit Records"
# http://www2.census.gov/acs2011_1yr/pums/csv_pus.zip
# http://www2.census.gov/acs2011_1yr/pums/csv_hus.zip`
# downloaded file details:
#   $ ls -lh *.zip
# 239M Oct 15 13:17 csv_hus.zip
# 580M Mar 4 06:31 csv_pus.zip
# $ shasum *.zip
# cdfdfb326956e202fdb560ee34471339ac8abd6c csv_hus.zip
# aa0f4add21e327b96d9898b850e618aeca10f6d0 csv_pus.zip


#Getting data from a MySQL Database
library(RMySQL)

#Create our driver
driver <- dbDriver("MySQL")

#Establish our connection
#con <- dbConnect(driver, dbname="Music_Demo",host="localhost",port=3306, 
#                  user="Josue",password="diciembre07")


if (mysqlHasDefault()) {
  con <- RMySQL::dbConnect(drv = driver, dbname = "Music_Demo",
                           user = "Josue", host = "localhost", port = 3306,
                           password = "diciembre07") 
  summary(con)
  dbGetInfo(con)
  dbListResults(con)
  dbListTables(con) #Show tables in the database
  dbDisconnect(con)
}

#Sends the query to MySQL and resturn the result of the statement
dAlbum <- dbGetQuery(conn = con , statement = "Select * from album")
dArtist <- dbGetQuery(conn = con, statement = "select * from artist")
dPlayed <- dbGetQuery(con , "select * from Played")
dTrack <- dbGetQuery(con , "select * from Track")

#Get the size of the data
object.size(dAlbum)

#Disconnect from the database
dbDisconnect(con)
save(dAlbum,dArtist,dPlayed,dTrack,file='Desktop/R Work/Learning/Practical Data Science With R/Data/database_sample.RData')

#To load the data
load(file = 'Desktop/R Work/Learning/Practical Data Science With R/Data/database_sample.RData')


#Get the PUMA data
load(file = "Desktop/R Work/Learning/Practical Data Science With R/Data/phsample.RData")
dhus
dpus

#Subsetting data according the following conditions
psub = subset(dpus,with(dpus,(PINCP>1000)&(ESR==1)&
                          (PINCP<=250000)&(PERNP>1000)&(PERNP<=250000)&
                          (WKHP>=40)&(AGEP>=20)&(AGEP<=50)&
                          (PWGTP1>0)&(COW %in% (1:7))&(SCHL %in% (1:24))))

#From 1 and 2 to M and F 
psub$SEX <- as.factor(as.character(ifelse(psub$SEX == 1, "M", "F")))
psub$SEX = relevel(psub$SEX, "M") #Reorders the levels placing M first

cowmap <- c("Employee of a private for-profit",
            "Private not-for-profit employee",
            "Local government employee",
            "State government employee",
            "Federal government employee",
            "Self-employed not incorporated",
            "Self-employed incorporated")

#Change from 1:7 to the list above
psub$COW <- as.factor(cowmap[psub$COW])
psub$COW = relevel(psub$COW,cowmap[1]) #Reorder the levels

schlmap = c(
  rep("no high school diploma",15),
  "Regular high school diploma",
  "GED or alternative credential",
  "some college credit, no degree",
  "some college credit, no degree",
  "Associate's degree",
  "Bachelor's degree",
  "Master's degree",
  "Professional degree",
  "Doctorate degree")

#Same for SCHl
psub$SCHL = as.factor(schlmap[psub$SCHL])
psub$SCHL = relevel(psub$SCHL,schlmap[1])

#Set train and test data
dtrain = subset(psub,ORIGRANDGROUP >= 500)
dtest = subset(psub,ORIGRANDGROUP < 500)

summary(dtrain$COW)


