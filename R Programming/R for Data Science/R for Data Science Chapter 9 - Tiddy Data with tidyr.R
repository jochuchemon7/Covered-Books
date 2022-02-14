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



############### Chapter 9 - Tiddy Data with tidyr

library(tidyverse)

#Different ways the data is presented
table1
table2
table3

table4a
table4b

#Tidy Data Set must be 
#1. Each variable must have its own column.
#2. Each observation must have its own row.
#3. Each value must have its own cell

#Here only table1 is tidy bc all variables have their own column 
table1

#Adds a columns name rate at the end
table1 %>%
  mutate(rate = cases / population * 10000)

#Compute cases per year using count and wt for what variable you want to add for
table1 %>%
  count(year, wt = cases)

table1 %>%
  count(year, wt = cases, population)

#PLot for the countries
ggplot(table1, aes(x = year,y = cases)) +
  geom_point(mapping = aes(color = country)) +
  geom_line(mapping = aes(group = country), color = "grey50")

######Spreading and Gathering

#Values of the variables are shown as the column varibles 
table4a



###Gathering
#When one variable is spread among many column names

#Using gather our initial colnames are set as values for key year and the values to 
#colname cases. Our gather columns are dropped
table4a %>%
  gather(`1999`, `2000`, key = "year", value = "cases")

#For table4b
table4b %>%
  gather(`1999`, `2000`, key = "year", value = "population")


tidy4a <- table4a %>%
  gather(`1999`, `2000`, key = "year", value = "cases")
tidy4b <- table4b %>%
  gather(`1999`, `2000`, key = "year", value = "population")

#Joins by country, year or by the same columns found in both
left_join(x = tidy4a,y = tidy4b)


#####Spreading
#When an observation is spread among rows

#Has a col name type that holds chr on cases & population and a count column
table2

#Takes the col type and gives a col for each value in there and the values 
#For those columns are count
spread(data = table2, key = type, value = count)


#####Separate
table3

#Using what ever separator character is in between the values
separate(data = table3, col = rate, into = c("cases", "population"))

#Or by indicating the separator character
table3 %>%
  separate(rate, into = c("cases", "population"), sep = "/")

#convert = TRUE  to change chr to whatever value suits best

###Unite
table5

table5 %>%
  unite(new, century, year)


##Missing Values
stocks <- tibble(
  year = c(2015, 2015, 2015, 2015, 2016, 2016, 2016),
  qtr = c( 1, 2, 3, 4, 2, 3, 4),
  return = c(1.88, 0.59, 0.35, NA, 0.92, 0.17, 2.66))
stocks

stocks %>%
  spread(year, return) %>%
  gather(year, return, `2015`:`2016`, na.rm = TRUE)


#################   Case Study
who

print(who, width = Inf)

#We take the columns and set then in a key column with the number of instances
who1 <- who %>%
  gather(new_sp_m014:newrel_f65, key = "key", value = "cases", na.rm = T)
who1


Mode   FALSE    TRUE 
logical    3173    4067 

#Count for each key the number of instances
who1 %>%
  count(key)

#We make key variables consistent for newrel -> new_rel like the rest
who2 <- who1 %>%
  mutate(key = stringr::str_replace(key, "newrel", "new_rel"))
who2

#We separate the new, type and sex into their own col since they are sep by `_`
who3 <- who2 %>%
  separate(key, c("new", "type", "sexage"), sep = "_")
who3

who3 %>%
  count(new)

#We remove the new and abbreviations of the countries
who4 <- who3 %>%
  select(-new, -iso2, -iso3)
who4

#We separate 
who5 <- who4 %>%
  separate(sexage, c("sex", "age"), sep = 1)
who5

#The whole thing 
who %>%
  gather(code, value, new_sp_m014:newrel_f65, na.rm = TRUE) %>%
  mutate(code = stringr::str_replace(code, "newrel", "new_rel")) %>%
  separate(code, c("new", "var", "sexage")) %>%
  select(-new, -iso2, -iso3) %>%
  separate(sexage, c("sex", "age"), sep = 1)










