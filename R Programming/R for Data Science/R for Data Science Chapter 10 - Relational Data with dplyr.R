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



############### Chapter 10 - Relational Data with dplyr

library(tidyverse)
#Contains 4 relational data sets
library(nycflights13)

#Getting the data frames or tibbles

flights <- nycflights13::flights
flights

airlines <- nycflights13::airlines
airlines

airports <- nycflights13::airports
airports

planes <- nycflights13::planes
planes

weather <- nycflights13::weather  
weather

#Looking for primary keys if that variable appears more than one (meaning not unique)
#If 0 rows then variable is a primary key
planes %>%
  count(tailnum) %>%
  filter(n > 1)

weather %>%
  count(year, month, day, hour, origin) %>%
  filter(n > 1)

flights %>%
  count(year, month, day, flight) %>%
  filter(n >1)

flights %>%
  count(year, month, day, tailnum) %>%
  filter(n >1)

####Mutating Joins

#Smaller variable data set
flights2 <- flights %>%
  select(year:day, hour, origin, dest, tailnum, carrier)
flights2

#We combine the name from the airlines to the flights2 by carrier
flights2 %>%
  select(-origin, -dest) %>%
  left_join(airlines, by = "carrier") #Automatically knows by carrier but just in case
flights2

#Same but with mutate
flights2 %>%
  select(-origin, -dest) %>%
  mutate(name = airlines$name[match(carrier, airlines$carrier)])


####Understanding Joins

#Sample Tables
x <- tribble(
  ~key, ~val_x,
  1, "x1",
  2, "x2",
  3, "x3")

y <- tribble(
  ~key, ~val_y,
  1, "y1",
  2, "y2",
  4, "y3")

x
y

###Inner Join: Observations that appear on both tables

#Gets all the columns for both data frames that have the same by key or
#`join by` value
inner_join(x,y)
#Or
x %>%
  inner_join(y, by = "key")


##Outer Joins: Observations that appear at least in one table

#Full for x key rows and NA for y (Uses key values of x)
left_join(x,y)

#Full for y key rows and NA for x (Uses key values of y)
right_join(x,y)

#Everything is combinesd NA for when key vals are different on y or x
full_join(x,y)


#left_join(x,y) has all of x and intercept from x & y 
#right_join(x,y) has all of y and intercept from y & x

####Duplicate Keys

#When 
x <- tribble(
  ~key, ~val_x,
  1, "x1",
  2, "x2",
  2, "x3",
  1, "x4")

y <- tribble(
  ~key, ~val_y,
  1, "y1",
  2, "y2")

left_join(x, y, by = "key")


x <- tribble(
  ~key, ~val_x,
  1, "x1",
  2, "x2",
  2, "x3",
  3, "x4")

y <- tribble(
  ~key, ~val_y,
  1, "y1",
  2, "y2",
  2, "y3",
  3, "y4")

left_join(x, y, by = "key")

#Natural join on variables found in both tables
flights2 %>%
  left_join(weather)

#Join only by tailnum
flights2 %>%
  left_join(planes, by = "tailnum")

#When they don´t have the same colname but they right values
flights2 %>%
  left_join(airports, by = c("dest" = "faa"))

#Now with origing from flights2 and not dest
flights2 %>%
  left_join(airports, c("origin" = "faa"))

#Prints the US map and a dot on each airport location 
airports %>%
  semi_join(flights, c("faa" = "dest")) %>%
  ggplot(mapping = aes(lon, lat)) +
  borders("state") +
  geom_point() +
  coord_quickmap()


#You can also use base:merge
merge(x,y all.x = T) #Left join


#Top ten destinations
top_dest <- flights %>%
  count(dest, sort = TRUE) %>%
  head(10)
top_dest

#We get the flights that went to that destination
flights %>%
  filter(dest %in% top_dest$dest)

left_join(top_dest, flights, by = "dest")

#Keeps the observations in flights that are found in top_dest
flights %>%
  semi_join(top_dest)

#Keeps rows from flights that match on dest in top_dest
semi_join(flights, top_dest, by = "dest")


##Anti_joins do the opposite and drop the rows that match
anti_join(flights, top_dest, by = "dest")

flights %>%
  anti_join(planes, by = "tailnum") %>%
  count(tailnum, sort = TRUE)


#1- To have clean data first identify the primary key vairiable. 
#Not always by their uniqueness but on the understanding of the data

#Ex altitude and longitude have unique identifiers but not good ones
airports %>% 
  count(alt, lon) %>%
  filter(n > 1)


#2- Make sure the primary key variables do not have a missing value
#3- Make sure the primary key match with the foreging key. Use anti_join()



######Set Operations

#Sample Data
df1 <- tribble(
  ~x, ~y,
  1, 1,
  2, 1)

df2 <- tribble(
  ~x, ~y,
  1, 1,
  1, 2)


#Return only observations in both df1 and df2.
intersect(df1, df2)

#Return unique observations in df1 and df2
union(df1, df2)

#Return observations in df1, but not in df2.
setdiff(df1, df2)
#Return observations in df2, but not in df1.
setdiff(df2,df1)


















