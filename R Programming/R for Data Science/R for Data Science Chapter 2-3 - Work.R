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




############### Chapter 2-3 - Workflow: Basics & Data Transformation

library(tidyverse)

#Filtering
ggplot(data = mpg) +
  geom_point(data = filter(mpg, cyl == 8), mapping = aes(x = displ, y = hwy),
             position = "jitter")



library(nycflights13)

data <- nycflights13::flights
filter(data, month == 1, day == 1)

#Using Logical 
filter(flights, month == 11 | month == 12)

nov_dec <- filter(flights, month %in% c(11, 12))
nov_dec

#Arrange the data the order of the `sort` ad if descending
arrange(data, year, month, day)
arrange(data, desc(dep_time)) #With descending on a column

#Using Select (NOT WORKING)
select(flights, year, month, day)
select(flights, year:day)
select(flights, -(year:day))

#Rename for an instance but assing to data for permanent change
rename(data, tail_num = tailnum)

#Plances the those 2 columns first in the select and the rest later (NOT WORKING)
select(flights, time_hour, air_time, everything())

select(flights, contains("TIME"))

#Select NOT WORKING
flights_sml <- select(flights, year:day, ends_with("delay"), distance, air_time)

#Adds those two columns at the end
mutate(flights_sml,
       gain = arr_delay - dep_delay,
       speed = distance / air_time * 60)

#For keeping only the new one
transmute(flights,
          gain = arr_delay - dep_delay,
          hours = air_time / 60,
          gain_per_hour = gain / hours)

x <- 1:10

#Cummulative actions
cumsum(x)        
cummean(x)       


#####Summarize()

summarize(data, delay = mean(dep_delay, na.rm = TRUE))

by_day <- group_by(flights, year, month, day)
summarize(by_day, delay = mean(dep_delay, na.rm = TRUE))


#Multiple Operations in The Pipe

by_dest <- group_by(flights, dest)
delay <- summarize(by_dest,
                   count = n(),
                   dist = mean(distance, na.rm = TRUE),
                   delay = mean(arr_delay, na.rm = TRUE))
delay <- filter(delay, count > 20, dest != "HNL")


par(mfrow = c(1, 2))

ggplot(data = delay, mapping = aes(x = dist, y = delay))+
  geom_point(mapping = aes(size = count), alpha = 1/3)+
  geom_smooth(color = "green", se = F)


# OR using the pipe operator
delays <- flights %>%
  group_by(dest) %>%
  summarize(
    count = n(),
    dist = mean(distance, na.rm = TRUE),
    delay = mean(arr_delay, na.rm = TRUE)
  ) %>%
  filter(count > 20, dest != "HNL")


#For missing values
flights %>%
  group_by(year, month, day) %>%
  summarize(mean = mean(dep_delay, na.rm = TRUE))


#Removing the na values/flights
not_cancelled <- flights %>%
  filter(!is.na(dep_delay), !is.na(arr_delay))

not_cancelled %>%
  group_by(year, month, day) %>%
  summarize(mean = mean(dep_delay))


######Counts

#Using geom_freqpoly()
delays <- not_cancelled %>%
  group_by(tailnum) %>%
  summarize(delay = mean(arr_delay))

ggplot(data = delays, mapping = aes(x = delay))+
  geom_freqpoly(binwidth = 10)


#Using geom_sacatter()
delays <- not_cancelled %>%
  group_by(tailnum) %>%
  summarize(delay = mean(arr_delay, na.rm = TRUE), n = n())

ggplot(data = delays, mapping = aes(x = n, y = delay)) +
  geom_point(alpha = 1/10)

#Filter to have the delays with more than 25 occurences
delays %>%
  filter(n > 25) %>%
  ggplot(mapping = aes(x = n, y = delay)) +
  geom_point(alpha = 1/10)


batting <- as_tibble(Lahman::Batting)

batters <- batting %>%
  group_by(playerID) %>%
  summarize(ba = sum(H, na.rm = TRUE) / sum(AB, na.rm = TRUE), 
            ab = sum(AB, na.rm = TRUE))

batters %>%
  filter(ab > 100) %>%
  ggplot(mapping = aes(x = ab, y = ba)) +
  geom_point() +
  geom_smooth(se = FALSE)

batters %>%
  arrange(desc(ba))

#####Useful Summary Function

not_cancelled %>%
  group_by(dest) %>%
  summarize(distance_sd = sd(distance)) %>%
  arrange(desc(distance_sd))

not_cancelled %>%
  group_by(year, month, day) %>%
  summarize(first = min(dep_time), 
            last = max(dep_time))


not_cancelled %>%
  group_by(year, month, day) %>%
  summarize(first_dep = first(dep_time), 
            last_dep = last(dep_time))

#For unique values n_distinct()
not_cancelled %>%
  group_by(dest) %>%
  summarize(carriers = n_distinct(carrier)) %>%
  arrange(desc(carriers))

#Counts of type of destination
not_cancelled %>%
  count(dest)

#You can assing wights to the count
not_cancelled %>%
  count(tailnum, wt = distance)

#Counts the number where the dep_time was less than 500
not_cancelled %>%
  group_by(year, month, day) %>%
  summarize(n_early = sum(dep_time < 500))


#A flight is 15 minutes early 50% of the time, and 15 minutes
#late 50% of the time. !DID NOT WORKED
not_cancelled %>%
  group_by(year, month, day) %>%
  summarize(n_early = mean(dep_time == 15), n_late = mean(dep_time == -15)) %>%
  filter(n_early == .5 && n_late == .5)





