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



############### Chapter 16 Vectors

library(tidyverse)

#Type
typeof(letters)
typeof(1:10)

#Length
x <- list("a", "b", 1:10)
length(x)


###Atomic Vectors

#Logical Vector
1:10 %% 3 == 0
c(TRUE, TRUE, FALSE, NA)

#Integer Vector
typeof(1L)

#Double Vector (floating point value/ approximation)
typeof(1)
c(-1, 0, 1) / 0 #Doubles can have special NA, NaN, Inf etc values

dplyr::near(2, sqrt(2)^2) #Near() when comparing against double values

#Character Vector

x <- "This is a reasonably long string."
pryr::object_size(x) #Byte size

y <- rep(x, 1000)
y
pryr::object_size(y) #Does not create duplicates but creates pointers to it

#NA valuesfor each atomic value
NA # logical
NA_integer_ # integer
NA_real_ # double
NA_character_ #character


##Coersion - conversion

#Implicit coersion
x <- sample(20, 100, replace = TRUE)
y <- x > 10
sum(y)
mean(y) 

#Adding vectors of different lengths
1:10 + 1:2

1:10 + 1:3 #Error no common multipler

#When using tibble use rep
tibble(x = 1:4, y = rep(1:2, 2))
tibble(x = 1:4, y = rep(1:2, each = 2))

#Naming vectors
c(x = 1, y = 2, z = 4)
set_names(c(1,2,3), c("a", "b", "c"))
set_names(1:3, c("a", "b", "c"))

#Subsetting
x <- c("one", "two", "three", "four", "five")
x[c(3, 2, 5)]
x[c(1, 1, 5, 5, 5, 2)]

#Nwgative drops the elements
x[c(-1, -3, -5)]

#Error no mixing neg and pos
x[c(1, -1)]

#All even numbser even NA´s
x <- c(10, 3, NA, 5, 8, 1, NA)
x[x %% 2 == 0]

#Subset with names
x <- c(abc = 1, def = 2, xyz = 5)
x[c("xyz", "def")]


##Lists
x <- list(1, 2, 3)
x
x <- list(a =1, b=2, c=3) #With names
str(x)

#More subsetting
a <- list(a = 1:3, b = "a string", c = pi, d = list(-1, -5))
a[[1]][2]
a[[4]][[1]]

#Attributes
x <- 1:10
x
attr(x, "greeting")
attr(x, "greeting") <- "Hi!"
attr(x, "farewell") <- "Bye!"
attributes(x)

#Look up methods
methods("as")
getS3method("as.Date", "default") #More detailed

#Factors
x <- factor(c("ab", "cd", "ab"), levels = c("ab", "cd", "ef"))
x
typeof(x)
attributes(x)

#Dates and Date-Times
x <- as.Date("1971-01-01")
x
unclass(x)
typeof(x)
attributes(x)

#With lubridate
x <- lubridate::ymd_hm("1970-01-01 01:00")
x
unclass(x)
typeof(x)
attributes(x)

#as.POSIXlt
y <- as.POSIXlt(x)
typeof(y)
attributes(y)

#Tibbles
x <- tibble(x =c(1,2,3), y = c("one", "two", "three"))
x
unclass(x)
typeof(x)
attributes(x)










