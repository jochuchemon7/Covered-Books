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


#####################Chapter 3 Exploring Data

#Donwload data from git 
#Josue$ curl LJO http://URL.com 
#To acces fo to finder and then cmd+shift+h and you will the file in your uer folder
custdata  <- read.table(file ="Desktop/R Work/Learning/Practical Data Science With R/Data/custdata.tsv", 
                        sep = "\t", header = TRUE)
head(custdata)
summary(custdata)

head(custdata$age)

#Histogram for age
ggplot(custdata, aes(x = age))+
  geom_histogram(binwidth = 5, color = "black", fill = "gray")

library(scales)

#Density for income
ggplot(custdata, aes(x = income)) +
  geom_density() +
  scale_x_continuous(labels = dollar) #Uses dollars as the x_continous label

#"Normalizesh" on log10 scale rather than continous (ignores 0 and negative values)
ggplot(data = custdata, aes(x =income)) +
  geom_density()+
  scale_x_log10(breaks = c(100,1000,10000,100000), 
                labels = dollar) +#Have the breaks on dollars
  annotation_logticks(sides = "bt") # Adds additional x axis annotations

#Using geom_bar for the marital status
ggplot(custdata, aes(x = marital.stat))+
  geom_bar(fill = "gray")

#Coord flip on our geom_bar for state of residence and theme y axes text size is .8
ggplot(custdata, aes(x = state.of.res)) +
  geom_bar(fill="gray") +
  coord_flip() +
  theme(axis.text.y=element_text(size=rel(0.8)))

####### Attempt for the code below
stateData <- custdata$state.of.res %>%
  count() %>%
  arrange(desc(freq))
colnames(stateData) <- c("state", "total")
ggplot(stateData) +
  geom_bar(aes(x = state, y = total), stat = "identity",fill = "gray")+
  coord_flip() 
########

#Counts the the occurences for each state
statesums <- table(custdata$state.of.res)
statef <- as.data.frame(statesums)
colnames(statef)<-c("state.of.res", "count")  

#Sort of arrange(desc(count))
statef <- transform(statef, state.of.res=reorder(state.of.res, count))
summary(statef)

#Plot
ggplot(statef, aes(x = state.of.res, y = count))+
  geom_bar(fill = "gray", stat = "identity")+
  coord_flip()+
  theme(axis.text.y = element_text(size = rel(.8)))


#Simple linear plot
x = runif(100)
y = x^2 + .2*x
ggplot(data.frame(x,y), aes(x=x,y=y))+
  geom_line()

#Examining data correlation
custdata2 <- subset(custdata,(custdata$age > 0 & custdata$age < 100 
                              & custdata$income > 0))
cor(custdata2$age, custdata2$income) #No correlation what so ever

#Plotting for more insight (not hard to see no correlation)
ggplot(custdata2, aes(x = age, y = income))+
  geom_point() +
  ylim(c(0,200000))

#Plot with the state_smoth with a linear model as the method
ggplot(custdata2, aes(x = age, y = income))+
  geom_point() +
  stat_smooth(method = "lm")+
  ylim(c(0,200000))

#Smoot line using the data point you can see where there is a rise and fall
ggplot(custdata2, aes(x = age, y = income)) +
  geom_point()+
  geom_smooth()+
  ylim(c(0,200000)) 

#Plot with 
ggplot(custdata2, aes(x=age, y=as.numeric(health.ins))) +
  geom_point(position=position_jitter(w=0.05, h=0.05)) +
  geom_smooth()

#Plots age vs has health insurace using a position_jitter(w =.05, h = .05)
ggplot(custdata2, aes(x = age, y = as.numeric(health.ins)))+
  geom_point(position = position_jitter(width = .05, height = .05))+
  geom_smooth()

#PLotting age vs income using geom_hex()
ggplot(custdata2, aes(x = age, y = income))+
  geom_hex()+
  ylim(c(0,200000)) +
  geom_smooth(colot = "white", se =F)

#Or change on the binwidth
ggplot(custdata2, aes(x=age, y=income)) +
  geom_hex(binwidth=c(5, 10000)) +
  geom_smooth(color="white", se=F) +
  ylim(0,200000)

#Geom_bar marital status and the count
ggplot(custdata2, aes(x = marital.stat)) +
  geom_bar(aes(fill = health.ins))

#Same but separates true and false health ins into own columns 
ggplot(custdata2, aes(x = marital.stat)) +
  geom_bar(aes(fill = health.ins), position = "dodge")

#Same but all columns have the same height count is on rate/proportion
ggplot(custdata2, aes(x = marital.stat)) +
  geom_bar(aes(fill = health.ins), position = "fill")

#Plot the bar from above with points at the bottom to get a sense of where you have
#The most data from
ggplot(custdata, aes(x=marital.stat)) +
  geom_bar(aes(fill=health.ins), position="fill") +
  geom_point(aes(y=-0.05), size=0.75, alpha=0.3, position=position_jitter(h=0.01))

#Plot with more caracteristics
ggplot(custdata2, aes(x = housing.type))+
  geom_bar(aes(fill = marital.stat), position = "dodge")+
  theme(axis.text.x = element_text(angle = 45, hjust =1))

#Plot and facet some of the characteristics
ggplot(custdata2, aes(x = marital.stat))+
  geom_bar(position = "dodge", fill = "darkgray")+
  facet_wrap(~housing.type, scales = "free_y")+ #Free_y gives y labes for each facet
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




















