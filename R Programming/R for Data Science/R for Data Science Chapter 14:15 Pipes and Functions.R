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



############### Chapter 14 - Pipes with magrittr

library(magrittr)
library(pryr)

#Set up
diamonds <- ggplot2::diamonds
diamonds2 <- diamonds %>%
  dplyr::mutate(price_per_carat = price / carat)
diamonds2

#Get sizes
pryr::object_size(diamonds) #Uses 3.46MB
pryr::object_size(diamonds2) #Uses 3.89MB
pryr::object_size(diamonds, diamonds2) #Both uses uses 3.89MB


diamonds$carat[1] <- NA #Change one value
pryr::object_size(diamonds)
pryr::object_size(diamonds2)
pryr::object_size(diamonds, diamonds2) #Creates a duplicate No longer 3.89 bu 4.32

###Using the Pipe

#Creates x variable with value of 10
assign("x", 10)
x

#assign() changes the value and tell the global enviroment
env <- environment()
"x" %>%
  assign(100, envir = env)
x

#Works
tryCatch(stop("!"), error = function(e) "An error")

#Error pipe should not be used
stop("!") %>%
  tryCatch(error = function(e) "An error")


###Other Tools with magrittr fails to srite the str bc of lack of T
rnorm(100) %>%
  matrix(ncol = 2) %>%
  plot() %>%
  str()

#Chain variable example, writtes str() at the end thanks to T
rnorm(100) %>%
  matrix(ncol = 2) %T>%
  plot() %>%
  str()

#Using $ rather than %>%
mtcars %$%
  cor(disp, mpg)

cor(mtcars$disp, mtcars$mpg)

#transform function to sort of update
mtcars <- mtcars %>%
  transform(cyl = cyl * 2)



############### Chapter 15 - Functions


df <- tibble::tibble(a = rnorm(10),
                     b = rnorm(10),
                     c = rnorm(10),
                     d = rnorm(10)) 

#Normalize data
df$a <- (df$a - min(df$a, na.rm = TRUE)) / (max(df$a, na.rm = TRUE) - min(df$a, na.rm = TRUE))
df$b <- (df$b - min(df$b, na.rm = TRUE)) / (max(df$b, na.rm = TRUE) - min(df$a, na.rm = TRUE))
df$c <- (df$c - min(df$c, na.rm = TRUE)) / (max(df$c, na.rm = TRUE) - min(df$c, na.rm = TRUE))
df$d <- (df$d - min(df$d, na.rm = TRUE)) / (max(df$d, na.rm = TRUE) - min(df$d, na.rm = TRUE))

#Using a single variable range[1]/[2] rather than min() and max()
rng <- range(df$a)
rng
(df$a - rng[1]) / (rng[2]- rng[1])

#Creating rescale function
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}
rescale01(c(0, 5, 10))

#With different inputs
rescale01(c(-10, 0, 10))
rescale01(c(1, 2, 3, NA, 5))

#Normalized simplified with the function
df$a <- rescale01(df$a)
df$b <- rescale01(df$b)
df$c <- rescale01(df$c)
df$d <- rescale01(df$d)

x <- c(1:10, Inf)
x
rescale01(x) #error bc of the Inf value

#We can fix the process by just changing a small part in the function
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE, finite = TRUE) #Fixed
  (x - rng[1]) / (rng[2] - rng[1])
}
rescale01(x) #Works


###Conditional Execution

has_name <- function(x) {
  nms <- names(x)
  if (is.null(nms)) {
    rep(FALSE, length(x)) 
  } 
  else { !is.na(nms) & nms != "" }
}

has_name(flights)

#Careful with comparing an integer and a numeric value
identical(0L, 0)


#Careful with floating point numbers
x <- sqrt(2) ^ 2
x
x == 2

#Use near when comparing for floating point numbers
dplyr::near(x,2)


###Checking Values

#Yoou can use stop() to throw an error when one happens
wt_mean <- function(x, w) {
  if (length(x) != length(w)) {
    stop("`x` and `w` must be the same length", call. = FALSE) #Sort of like break()
  }
  sum(w * x) / sum(x)
}

wt_mean(1:10,1:5) #Throws error for not having the same lenghts


wt_mean <- function(x, w, na.rm = FALSE) {
  stopifnot(is.logical(na.rm), length(na.rm) == 1) #Stop if not all true
  stopifnot(length(x) == length(w)) #Stop if not all true
  
  if (na.rm) {
    miss <- is.na(x) | is.na(w)
    x <- x[!miss]
    w <- w[!miss]
  }
  sum(w * x) / sum(x)
}
wt_mean(1:6, 6:1, na.rm = "foo")

###Dot, Dot, Dot

#You can use ... to take multiple values without needing to define the argument
commas <- function(...) stringr::str_c(..., collapse = ", ")
commas(letters[1:10])


#You can use the return() function to return values early in the function


#Using the invisible() does not print the passed argument
show_missings <- function(df) {
  n <- sum(is.na(df))
  cat("Missing values: ", n, "\n", sep = "")
  invisible(df)
}

show_missings(mtcars)
x <- show_missings(mtcars)
#However it is still there but not printed
class(x)
dim(x)


#And we can still use a pipe (Here we compute the show missing twice)
mtcars %>%
  show_missings() %>%
  mutate(mpg = ifelse(mpg < 20, NA, mpg)) %>%
  show_missings()





