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



############### Chapter 12 - Factors with forcast

library(tidyverse)
library(forcats)

###Creating Factors

#Create a list (class is character)
x1 <- c("Dec", "Apr", "Jan", "Mar")
#sorts in alphabetical order
sort(x1)

#Create the levels (sorted in order)
month_levels <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#Create an actual factor and stores the levels
y1 <- factor(x = x1, levels = month_levels)
y1
sort(y1) #Sorts based on the order of the levels

#Any value not found in the levels would be replaced by NA
x2 <- c("Dec", "Apr", "Jam", "Mar")
y2 <- factor(x2, levels = month_levels)
y2

#If you want the order of the levels in the order of values in the vector (unique())
f1 <- factor(x1, levels = unique(x1))
f1

#Data
gss_cat <- gss_cat

gss_cat %>%
  count(race)

ggplot(data = gss_cat, mapping = aes(x = race)) +
  geom_bar()+
  scale_x_discrete(drop = FALSE) #To include Not applicable since has 0 values


###Madifying Factor Order

relig <- gss_cat %>%
  group_by(relig) %>%
  summarize(age = mean(age, na.rm = T),
            tvhours = mean(tvhours, na.rm = T,),
            n = n())


ggplot(data = relig, mapping = aes(x = tvhours, y = relig)) +
  geom_point()

#Orders the religions by tvhours
ggplot(data = relig, mapping = aes(x = tvhours, y = fct_reorder(relig, tvhours))) +
  geom_point()

#Can rewrite like this as well
relig %>%
  mutate(relig = fct_reorder(relig, tvhours)) %>%
  ggplot(aes(tvhours, relig)) +
  geom_point()

#Same as above but for rincome
rincome <- gss_cat %>%
  group_by(rincome) %>%
  summarize(age = mean(age, na.rm = T),
            tvhours = mean(tvhours, na.rm = T),
            n = n())

#Not good idea since rincome are already has a principled order 
ggplot(data = rincome, aes(x = age, fct_reorder(rincome, age)))+
  geom_point()

#Here we place `Not applicable` first 
ggplot(rincome, aes(x = age, y = fct_relevel(rincome, "Not applicable"))) +
  geom_point()

#By age
by_age <- gss_cat %>%
  filter(!is.na(age)) %>%
  group_by(age, marital) %>%
  count() %>%
  mutate(prop = n/sum(n))

#Plot
ggplot(by_age, aes(x = age, y = prop, color = marital)) +
  geom_line(na.rm = T)

gss_cat %>%
  mutate(marital = marital %>%
           fct_infreq() %>% #Incremental count order
           fct_rev()) %>% #From low to high
  ggplot(aes(marital)) +
  geom_bar()

###Modifying Factor Levels
gss_cat %>%
  count(partyid)

#We changed the levels on partyid factor column
gss_cat %>%
  mutate(partyid = fct_recode(partyid,
                              "Republican, strong" = "Strong republican",
                              "Republican, weak" = "Not str republican",
                              "Independent, near rep" = "Ind,near rep",
                              "Independent, near dem" = "Ind,near dem",
                              "Democrat, weak" = "Not str democrat",
                              "Democrat, strong" = "Strong democrat")) %>%
  count(partyid)


#You can also combine multiple past levels into one
gss_cat %>%
  mutate(partyid = fct_recode(partyid,
                              "Republican, strong" = "Strong republican",
                              "Republican, weak" = "Not str republican",
                              "Independent, near rep" = "Ind,near rep",
                              "Independent, near dem" = "Ind,near dem",
                              "Democrat, weak" = "Not str democrat",
                              "Democrat, strong" = "Strong democrat",
                              "Other" = "No answer",
                              "Other" = "Don't know",
                              "Other" = "Other party")) %>%
  count(partyid)


#You can collapse multiple levels into lesser ones using fct_collapse()
gss_cat %>%
  mutate(partyid = fct_collapse(partyid,
                                other = c("No answer", "Don't know", "Other party"),
                                rep = c("Strong republican", "Not str republican"),
                                ind = c("Ind,near rep", "Independent", "Ind,near dem"),
                                dem = c("Not str democrat", "Strong democrat"))) %>%
  count(partyid)

#Lumps all levels expect the most common one using 
gss_cat %>%
  mutate(relig = fct_lump(relig)) %>%
  count(relig)

#Same but leaves the top 10 most common levels
gss_cat %>%
  mutate(relig = fct_lump(relig, n = 10)) %>%
  count(relig, sort = TRUE) %>%
  print(n = Inf)




