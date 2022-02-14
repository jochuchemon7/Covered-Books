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



############### Chapter 20 Many Models with purrr and broom


library(modelr)
library(tidyverse)

#Data Set
library(gapminder)
gapminder

#Plot information (Hard to see what is going on)
gapminder %>%
  ggplot(aes(year, lifeExp, group = country)) +
  geom_line(alpha = 1/3)


nz <- filter(gapminder, country == "New Zealand")
nz

#Plotting New Zeland lifeExp on the year
nz %>% 
  ggplot(aes(year, lifeExp)) +
  geom_line() +
  ggtitle("Full data = ")

nz_mod <- lm(lifeExp ~ year, data = nz) #Model
summary(nz_mod)

#Plot the prediction (Very Linear)
nz %>%
  add_predictions(nz_mod) %>%
  ggplot(aes(year, pred)) +
  geom_line() +
  ggtitle("Linear trend + ")

#Plot the residuals
nz %>%
  add_residuals(nz_mod) %>%
  ggplot(aes(year, resid)) +
  geom_hline(yintercept = 0, color = "white", size = 3) +
  geom_line() +
  ggtitle("Remaining pattern")


###Nested Data
by_country <- gapminder %>%
  group_by(country, continent) %>%
  nest() #We nest the year, lifeExp, pop and gdpPercap for each row on contry & continent
by_country

#Example to see the data for one country
by_country$data[[1]]


###List Columns

#Functions so we can take each df for each country
country_model <- function(df) {
  lm(lifeExp ~ year, data = df)
}

#Model on each data 
models <- map(by_country$data, country_model)
models
summary(models[[1]]) #Summary for a model applied on a country

#Group by Model
by_country <- by_country %>%
  mutate(model = map(data, country_model))
by_country

#Filter by Europe
by_country %>%
  filter(continent == "Europe")

#Sort by continent and then by country
by_country %>%
  arrange(continent, country)



###UnNesting

#Add the residual to all the models and data for each country
by_country <- by_country %>%
  mutate(resids = map2(data, model, add_residuals))
by_country

#Unnest the residuals form the data frame for each observation on the data 
resids <- unnest(by_country, resids)
resids

#Plot the residual by year and group by country
resids %>%
  ggplot(aes(year, resid)) +
  geom_line(aes(group = country), alpha = 1 / 3) +
  geom_smooth(se = FALSE)

#Plot same but with facet on continent
resids %>%
  ggplot(aes(year, resid)) +
  geom_line(aes(group = country), alpha = 1 / 3) +
  facet_wrap(~continent)
#Large residuals on africa later years- meaning our model is missing inst fitting
#too well


###Model Quality

#To see some model metrics
broom::glance(nz_mod)

#UnNest by glance on the models
by_country %>%
  mutate(glance = map(model, broom::glance)) %>%
  unnest(glance)

#To drop anything thant is not part of the metrics from broom::glance
glance <- by_country %>%
  mutate(glance = map(model, broom::glance)) %>%
  unnest(glance, .drop = TRUE)
glance

#Sort by r.square foe each model per country
glance %>%
  arrange(r.squared)

#Plot the bad models with the r.square vs continents
glance %>%
  ggplot(aes(continent, r.squared)) +
  geom_jitter(width = 0.5)

#Same as above
ggplot(glance, aes(continent, r.squared))+
  geom_jitter(width = .5)

#Get countries that have a bad r.squared
bad_fit <- filter(glance, r.squared < 0.25)

#Plot them
gapminder %>%
  semi_join(bad_fit, by = "country") %>% #Join for countrys from bad_fit found in gapminder
  ggplot(aes(year, lifeExp, color = country)) +
  geom_line()


###List Columns


#data
data.frame(x = list(1:3, 3:5))

#Puts a list in one of the columns
data.frame(x = I(list(1:3, 3:5)), #Using I()
           y = c("1, 2", "3, 4, 5"))

#Using tibble
tibble(x = list(1:3, 3:5),
       y = c("1, 2", "3, 4, 5"))


###Creating List Columns


#Using nest()
Nest1 <- gapminder %>%
  group_by(country, continent) %>% #Group by country and continent (Also keeps them)
  nest()

#Using ungrouped nest and defining what to nest
Nest2 <- gapminder %>%
  nest(year:gdpPercap)


###From Vectorized Functions

#Data
df <- tribble(  ~x1,
                "a,b,c",
                "d,e,f,g")
df

#You str_split x1 into a new x2 vector column
df %>%
  mutate(x2 = stringr::str_split(x1, ","))

#Now UnNested by each observation/(character) form x2
df %>%
  mutate(x2 = stringr::str_split(x1, ",")) %>%
  unnest()

#List inside a tibble 
sim <- tribble(
  ~f, ~params,
  "runif", list(min = -1, max = -1),
  "rnorm", list(sd = 5),
  "rpois", list(lambda = 10))
sim

#Invoke map form a function to the params and 10 times
sim %>%
  mutate(sims = invoke_map(f, params, n = 10))



###From Multivalued Summaries

#summarize can only work with summary functions that return a single value
mtcars %>%
  group_by(cyl) %>%
  summarize(q = quantile(mpg)) #quantile does not work

#But you can have th results in a list
mtcars %>%
  group_by(cyl) %>%
  summarize(q = list(quantile(mpg)))

#Or capture th probabilities
probs <- c(0.01, 0.25, 0.5, 0.75, 0.99)

#sort of data_grid for the list of probabilities and the quantiles for the mpgs per cyl
mtcars %>%
  group_by(cyl) %>%
  summarize(p = list(probs), q = list(quantile(mpg, probs))) %>%
  unnest()


###From Named List 

#Data
x <- list(
  a = 1:5,
  b = 3:4,
  c = 5:6)
x

#enframe- from a list of values a tibble table with values col regardless the size
df <- enframe(x)
df

#Now if you want to iterate over names and values in parallel, you can use map2()
df %>%
  mutate(smry = map2_chr(name, value, ~ stringr::str_c(.x, ": ", .y[1])))
df


###Simplifying List Columns

##List to Vector
df <- tribble(
  ~x,
  letters[1:5],
  1:3,
  runif(5))
df$x

#We "unnest" for a single value of col and give type and length using map_chr/int
df %>% 
  mutate(type = map_chr(x, typeof), length = map_int(x, length))

#New Data
df <- tribble(~x,
              list(a = 1, b = 2),
              list(a = 2, c = 4))
df

#.null = NA_real_ for lgl avoid error when no b element is found in a list of x
df %>% mutate(
  a = map_dbl(x, "a"),
  b = map_dbl(x, "b", .null = NA_real_)) 
df


###UnNesting

#x must have the same number of values that are in the list
tibble(x = 1:2, y = list(1:4, 1)) %>% 
  unnest(y)


Sample <- tibble(x = 1:3, y = list(1:3, 1, 2))
Sample$y
Sample %>%
  unnest()

#Values for y and z based on their position are the same length
df1 <- tribble(
  ~x, ~y, ~z,
  1, c("a", "b"), 1:2,
  2, "c", 3)
df1
df1 %>%
  unnest()

df1 <- tribble(
  ~x, ~y, ~z,
  1,"a", 1:2,
  2,c("b","c"), 3)
df1
df1 %>%
  unnest(y, z)#Bc y and z have different number of elements it fails


















