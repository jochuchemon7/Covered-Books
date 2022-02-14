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



############### Chapter 7/8 Tibbles with Tibble & Data Import with readr

library(tidyverse)

data <- iris
#From data frame to tibble
data <- as_tibble(data)

#Create your own
tibble(x = 1:5, y = 1, z = x ^ 2 + y)

#To refer to unusal colnames use the `` character
tb <- tibble(
  `:)` = "smile",
  ` ` = "space",
  `2000` = "number")
tb

#You can desing the table using tribble (for transpose tibble)
tribble(
  ~x, ~y, ~z,~f,
  #--|--|----|----
  "a", 2, 3.6, 123,
  "b", 1, 8.5, 123
)

#Does not display everything but 10 rows for every col
tibble(
  a = lubridate::now() + runif(1e3) * 86400,
  b = lubridate::today() + runif(1e3) * 30,
  c = 1:1e3,
  d = runif(1e3),
  e = sample(letters, 1e3, replace = TRUE))

#To display all of the columns and 10 rows
print(flights, n = 10 , width = Inf)


#####Subsetting
df <- tibble(
  x = runif(5),
  y = rnorm(5))

#They are all the same
df$x
df[["x"]]
df[[1]]

#For %>% use a dot for reference
df %>% .$x


######################### Chapter 8 Data Import with readr

library(tidyverse)

#Tibble example read_csv if ro tibble while read.csv is for data.frame
read_csv("a,b,c
         1,2,3
         4,5,6")

#To skip the first 2 lines in the csv file
read_csv("The first line of metadata
         The second line of metadata
         x,y,z
         1,2,3", skip = 2)

#Same for comments
read_csv("# A comment I want to skip
         x,y,z
         1,2,3", comment = "#")

#Without Assingning colnames to not treat the first row as a the colnames
read_csv("1,2,3\n4,5,6", col_names = FALSE)

#Or pass them the colnames
read_csv("1,2,3\n4,5,6", col_names = c("x", "y", "z"))

#Treats dot values as NA
read_csv("a,b,c\n1,2,.", na = ".")

#Use fread for quick reads


#######Pasing a Vector

##From character to any other data type

#From character to logical values
str(parse_logical(c("TRUE", "FALSE", "NA")))

#Same for integers
str(parse_integer(c("1", "2", "3")))

#And date
str(parse_date(c("2010-01-01", "1979-10-14")))

#For dealing with dots have them as na values
parse_integer(c("1", "231", ".", "456"), na = ".")

#Warning if parsing fails
x <- parse_integer(c("123", "345", "abc", "123.45"))

#Using problems you can see the expect vs actual given value on which row
problems(x)

#We can change the decimal markForm , to .
parse_double("1,23", locale = locale(decimal_mark = ","))

#To just get the number and ignore the rest 
parse_number("$100")
parse_number("It cost $123.45")
parse_number("20%")


####### Strings

#Representation of Strings each hexadecimal num is a byte of each character on it
charToRaw("Hadley")


#####Factors
fruit <- c("apple", "banana")

#Error since bananana is not found in the levels of fruit
parse_factor(c("apple", "banana", "bananana"), levels = fruit)

####Dates, Date-Times, and Times

#Using the international standard (form biggest to lowest)
parse_datetime("2010-10-01T201000")

# If time is omitted, it will be set to midnight
parse_datetime("2010-10-10")

#pase_date expects that specific format
parse_date("2010-10-01")

#For parse_time (Hour:minute:second)
parse_time("10:14:32")

#You can create your own formats
parse_date("01/02/15", "%m/%d/%y")
parse_date("01/02/15", "%d/%m/%y")
parse_date("01/02/15", "%y/%m/%d")

#You can also guess the type of data type 
guess_parser("2010-10-01")
guess_parser(c("TRUE", "FALSE"))

#Issues (Example of using guess_max of rows to use)
challenge <- read_csv(readr_example("challenge.csv"), guess_max = 1001)
challenge

#From triblle with chr to guessing their data types
df <- tribble(
  ~x, ~y,
  "1", "1.21",
  "2", "2.32",
  "3", "4.56"
)
df
df2 <- type_convert(df)
df2


#####Write to a File
write_csv(challenge, "challenge.csv")

#Implements a binary formart of storage
library(feather)
write_feather(challenge, "challenge.feather")
read_feather("challenge.feather")





















