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




############### Chapter 4-5 Explorational Data Analysis

#On sets of .5 lenght
diamonds %>%
  count(cut_width(carat, 0.5))

ggplot(data = diamonds, mapping = aes(x = carat)) +
  geom_histogram(binwidth = .5)


#With different smaller bindwith for smaller sections and in turn more 
#There takes only those with carat smaller than 3
smaller <- diamonds %>%
  filter(carat < 3)

ggplot(data = smaller, mapping = aes(x = carat)) +
  geom_histogram(binwidth = 0.1)


#For multiple hitograms at once (geom_freqpoly)
ggplot(data = smaller, mapping = aes(x = carat, color = cut)) +
  geom_freqpoly(binwidth = 0.1)


ggplot(data = faithful, mapping = aes(x = eruptions))+
  geom_histogram(binwidth = .5)

#Large y range
ggplot(diamonds) +
  geom_histogram(mapping = aes(x = y))
summary(diamonds$y) #Mean 5.7 but max 58

ggplot(diamonds) +
  geom_histogram(mapping = aes(x = y), binwidth = .5) +
  coord_cartesian(ylim = c(0,50))

unusual <- diamonds %>%
  filter(y < 3 | y > 20) %>%
  arrange(y)
unusual

#We cust out the outliers
diamonds2 <- diamonds %>%
  filter(between(y, 3, 20))
diamonds2

#Or replace the outliers with na values
diamonds2 <- diamonds %>%
  mutate(y = ifelse(y < 3 | y > 20, NA, y)) #(Test, if true, if false)

#We can see the na instances for y
diamonds2 %>%
  filter(is.na(y))

#For na values, they are removed
ggplot(diamonds2, mapping = aes(x =x, y =y))+
  geom_point()

#Or supprese the warning
ggplot(data = diamonds2, mapping = aes(x = x, y = y)) +
  geom_point(na.rm = TRUE)

#If you want to understand what makes observations with missing values different
#from observations with recorded values. 
nycflights13::flights %>%
  mutate(
    cancelled = is.na(dep_time),
    sched_hour = sched_dep_time %/% 100,
    sched_min = sched_dep_time %% 100,
    sched_dep_time = sched_hour + sched_min / 60
  ) %>%
  ggplot(mapping = aes(x = sched_dep_time)) +
  geom_freqpoly( mapping = aes(color = cancelled), binwidth = 1/4)

##Covariation
#With count as y axis
ggplot(data = diamonds, mapping = aes(x = price)) +
  geom_freqpoly(mapping = aes(color = cut), binwidth = 500)

#Instead of displaying the count we do the density 
ggplot(data = diamonds, mapping = aes(x = price, y = ..density..)) +
  geom_freqpoly(mapping = aes(color = cut), binwidth = 500)

#Using a box plot
ggplot(data = diamonds, mapping = aes(x = cut, y = price)) +
  geom_boxplot()


#With the mpg data set
ggplot(data = mpg, mapping = aes(x = class, y = hwy))+
  geom_boxplot()

#We can reorder the levels based on their values (thi case by the median)
ggplot(data = mpg) +
  geom_boxplot(
    mapping = aes(
      x = reorder(class, hwy, FUN = median),
      y = hwy)
  )

#For long class names
ggplot(data = mpg) +
  geom_boxplot(
    mapping = aes(
      x = reorder(class, hwy, FUN = median),
      y = hwy)) +
  coord_flip()

####Two Categorical Values

#Using shapes for the count on categorical variables
ggplot(data = diamonds) +
  geom_count(mapping = aes(x = cut, y = color))

#Using the count function (Same w/o the graph)
diamonds %>%
  count(color, cut)

#Visualize with geom_tile and use color depthness to represent the count (fill)
diamonds %>%
  count(color, cut) %>%
  ggplot(mapping = aes(x = color, y = cut)) +
  geom_tile(mapping = aes(fill = n))


###Two Continous Variables 

#Plots overlap
ggplot(data = diamonds) +
  geom_point(mapping = aes(x = carat, y = price))

#Solution with alpha
ggplot(data = diamonds) +
  geom_point(mapping = aes(x = carat, y = price), alpha = 1/100)

#Using geom_bind2d (Uses the geom_tile for the count) Avoids overlapping on point
ggplot(data = smaller) +
  geom_bin2d(mapping = aes(x = carat, y = price))

#Using geom_hex() (Similar to geom_bind2d) Avoids overlapping on points
ggplot(data = smaller) +
  geom_hex(mapping = aes(x = carat, y = price))


######Patterns and Models

#To see for patterns 
ggplot(data = faithful) +
  geom_point(mapping = aes(x = eruptions, y = waiting))

library(modelr)

mod <- lm(log(price) ~ log(carat), data = diamonds)

diamonds2 <- diamonds %>%
  add_residuals(mod) %>%
  mutate(resid = exp(resid))

ggplot(data = diamonds2) +
  geom_point(mapping = aes(x = carat, y = resid))

ggplot(data = diamonds2) +
  geom_boxplot(mapping = aes(x = cut, y = resid))




