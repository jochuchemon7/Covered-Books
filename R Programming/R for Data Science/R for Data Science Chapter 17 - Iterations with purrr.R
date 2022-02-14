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



############### Chapter 17 Iterations with purrr

library(tidyverse)

###For loops
df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10))

#Repetitive
median(df$a)
median(df$b)
median(df$c)
median(df$d)

#For loop for calc median
output <- vector("double", ncol(df))
output
for (i in seq_along(df)) { # 2. sequence
  output[[i]] <- median(df[[i]]) # 3. body
}
output

#Example with mtcars
output <- vector("double", ncol(mtcars)) #Need to store the outputs
output
for (i in seq_along(mtcars)) { #seq_along() gives numeric sequence
  output[[i]] = mean(mtcars[[i]])
}
output

##Modifying existing object

df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10))
df

rescale01 <- function(x) {
  rng <- range(x, na.rm = T)
  (x - rng[1]) / (rng[2] - rng[1])
}

#Function rathern than calling 
for (i in seq_along(df)) {
  df[[i]] <- rescale01(df[[i]])
}
df

##Looping Patterns

x <- mtcars
results <- vector("list", length(x))
results
names(results) <- names(x)
results

for (i in seq_along(x)) {
  name <- names(x)[[i]]
  value <- x[[i]]
}


###Unkown output length

means <- c(0, 1, 2)
means

#You can just save the result in a list and then combined into a single vector
#when the loop is done
out <- vector("list", length(means))
out
for (i in seq_along(means)) {
  n <- sample(100, 1)
  out[[i]] <- rnorm(n, means[[i]])
}
str(out)
out[1]
str(unlist(out))


##More on when unknown sequence length
flip <- function() sample(c("T", "H"), 1)
flips <- 0
nheads <- 0
flip
while (nheads < 3) {
  if (flip() == "H") {
    nheads <- nheads + 1
  } else {
    nheads <- 0
  }
  flips <- flips + 1
}
flips


#Consider
df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10))

#Compute mean for every col using a for loop
output <- vector("double", ncol(df))
for (i in seq_along(df)) {
  output[[i]] = mean(df[[i]])
}
output

#Using a function to calculate to any tibble data frame
col_mean <- function(df) {
  output <- vector("double", length(df))
  for (i in seq_along(df)) {
    output[i] <- mean(df[[i]])
  }
  output
}
col_mean(df)

##Duplication for meadian and standard deviation
#Function for median
col_median <- function(df) {
  output <- vector("double", length(df))
  for (i in seq_along(df)) {
    output[i] <- median(df[[i]])
  }
  output
}
#Function for standard deviation
col_sd <- function(df) {
  output <- vector("double", length(df))
  for (i in seq_along(df)) {
    output[i] <- sd(df[[i]])
  }
  output
}

col_mean(df)
col_median(df)
col_sd(df)


#Using a single function for multiple actions 
col_summary <- function(df, fun) {
  out <- vector("double", length(df))
  for (i in seq_along(df)) {
    out[[i]] = fun(df[[i]])
  }
  out
}

#Using df and the fun
col_summary(df, mean)
col_summary(df, sd)
col_summary(df, median)


###Map Functions

#Take vector and function, return vector of same length and applies the function
#Makes a double vector 
map_dbl(df, median)
map_dbl(df, mean)
map_dbl(df, sd)

#Using pipe chain
df %>%
  map_dbl(sd)

#map() has ... arguments
map_dbl(df, mean)
map_dbl(df, mean, trim = .5) 

#It also perserves names
z <- list(x = 1:3, y = 4:5)
z
map_int(z, length)


###Short Cuts

#Using the split(.$cyl) it splits the data based on the number of cylinders
models <- mtcars %>%
  split(.$cyl) %>%
  map(function(df) lm(mpg ~ wt, data = df))
models

#You can also do
models <- mtcars %>%
  split(.$cyl) %>%
  map(~lm(mpg ~ wt, data = .)) #· refers to the list elements
models

#Apply summary and then extract the r.square from the summary on each model
models %>%
  map(summary) %>%
  map_dbl(~.$r.squared)  # ~ and then the function to be applied as

#You can even used a string (r.square) on each cylinder summary
models %>%
  map(summary) %>%
  map_dbl("r.squared")


x <- list(list(1, 2, 3), list(4, 5, 6), list(7, 8, 9))
x
x %>% map_dbl(2) #You can select the elements by position


#sapply and lapply are the base functions that act like map
x1 <- list(
  c(0.27, 0.37, 0.57, 0.91, 0.20),
  c(0.90, 0.94, 0.66, 0.63, 0.06),
  c(0.21, 0.18, 0.69, 0.38, 0.77))

x2 <- list(
  c(0.50, 0.72, 0.99, 0.38, 0.78),
  c(0.93, 0.21, 0.65, 0.13, 0.27),
  c(0.39, 0.01, 0.38, 0.87, 0.34))

x1
x2

threshold <- function(x, cutoff = 0.8) 
  x[x > cutoff]

x1 %>%
  sapply(threshold) %>% 
  str()


###Dealing with Failure

#Using safely()
safe_log <- safely(log)
safe_log()
str(safe_log(10)) #Returns error: NULL if succesfull else 
str(safe_log("a")) #result is null and an message is given at the error part

#Working with map()
x <- list(1, 10, "a")
x
y <- x %>%  #Using map to apply safely on each elemnt on our list
  map(safely(log))
y
str(y) #2 Succeses adn one failure

#transpose
y <- y %>% purrr::transpose() #Using transpose successes and failures are split
str(y)

#Using possibly you can check for succeses and failures if you get an NA value
x <- list(1, 10, "a")
x
x %>%
  map_dbl(possibly(log, NA_real_))  #Failure returns a NA value

#Quietly
x <- list(1, -1)
x
x %>%
  map(quietly(log)) %>% #Captures printed output, messages, and warnings
  str()



###Mapping Over Multiple Arguments

#Using
mu <- list(5, 10, -3) #The different means
mu
mu %>%
  map(rnorm, n = 5) %>% #Gives the n = 5 arument to rnorn for num of observations
  str()

#Same as
map(mu, ~ rnorm(n=5)) #You just fill in the parameters

#If you want to iterate thru the sd and means
sigma <- list(1, 5, 10)
sigma
seq_along(mu) %>% #seq_along index on mu
  map(~rnorm(5, mu[[.]], sigma[[.]])) %>% #Using mu and sigma iterate their indices
  str()                                   #And pass their respective values


#Instead use map2()
map2(mu, sigma, rnorm, n = 5) %>%  #Pass the two vectors and then the function
  str()


n <- list(1, 3, 5)
n
args1 <- list(n, mu, sigma)
args1
args1 %>%
  pmap(rnorm) %>% #pmap 
  str()

#Data
params <- tribble(
  ~mean, ~sd, ~n,
  5, 1, 1,
  10, 5, 3,
  -3, 10, 5)

params %>% 
  pmap(rnorm) #Using pmap() to apply a function on the tibble


f <- c("runif", "rnorm", "rpois")
f
param <- list(
  list(min = -1, max = 1),
  list(sd = 5),
  list(lambda = 10))
param
invoke_map(f, param, n = 5) %>% str() #Passes the list of functions and the vector
#of numbers to test with and the n = 5 observ
#first list goes with first f string and second with the second and so on 


#
sim <- tribble(
  ~f, ~params,
  "runif", list(min = -1, max = 1),
  "rnorm", list(sd = 5),
  "rpois", list(lambda = 10))
sim
sim %>%
  mutate(sim = invoke_map(f, params, n = 10))


###Walk

#You can see more of the action thant the returned value
x <- list(1, "a", 3)
x
x %>%
  walk(print)


library(ggplot2)
plots <- mtcars %>%
  split(.$cyl) %>%
  map(~ggplot(., aes(mpg, wt)) + geom_point())
plots
paths <- stringr::str_c(names(plots), ".pdf")
paths
pwalk(list(paths, plots), ggsave, path = tempdir())


###Predicate Functions

#Those that are factors are keep
iris %>%
  keep(is.factor) %>% #Keeps elements where the predicate is TRUE or FALSE  
  str()

#Those who are not factors are discarded
iris %>%
  discard(is.factor) %>%
  str()


# Some returns TRUE or FALSE if the condition is meet
x <- list(1:5, letters, list(10))
x
x %>%
  some(is_character)
x %>%
  every(is_vector)


#Detect returns the first value that complies with the predicate/condition
x <- sample(10)
x
x %>%
  detect(~ . > 5)
x %>%
  detect_index(~ . > 5) #To get the index of it


###Reduce and Accumulate

dfs <- list(
  age = tibble(name = "John", age = 30),
  sex = tibble(name = c("John", "Mary"), sex = c("M", "F")),
  trt = tibble(name = "Mary", treatment = "A"))
dfs

dfs %>% 
  reduce(full_join) #Joint columns and join them by name

vs <- list(
  c(1, 3, 5, 6, 10),
  c(1, 2, 3, 7, 8, 10),
  c(1, 2, 3, 4, 8, 9, 10))
vs
vs %>% reduce(intersect)



