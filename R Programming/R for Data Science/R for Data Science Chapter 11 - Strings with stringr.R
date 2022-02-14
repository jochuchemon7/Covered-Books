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



############### Chapter 11 - Strings with stringr

library(tidyverse)
library(stringr)

string1 <- "This is a string"
string2 <- 'To put a "quote" `quote` inside a string, use single quotes'

string1
string2

double_quote <- "\"" 
single_quote <- '\'' 

double_quote
single_quote

example <- "\`hello`"
example
#To see without the ""
writeLines(example)

#Non English characters
x <- "\u00b5"
x

#Character vector
c("one", "two", "three")

###String lenght
str_length(c("a", "R for data science", NA))
str_length("potatoe")

###Combining Strings
str_c("one", "two")
str_c("one", "two", "three", sep = " ")

#To print the NA literal
x <- c("abc", NA)
str_c("|-", str_replace_na(x), "-|")
str_replace_na(x)

#You can place before/after each string
str_c("prefix-", c("a", "b", "c"), "-suffix")

#Using if on str_c
name <- "Hadley"
time_of_day <- "morning"
birthday <- FALSE
str_c(
  "Good ", time_of_day, " ", name,
  if (birthday) " and HAPPY BIRTHDAY", ".")

#Collapse a string
str_c(c("x", "y", "z"), collapse = ", ")

#Subsetting Strings
x <- c("Apple", "Banana", "Pear")
str_sub(x, 1, 3) #Start, end
str_sub(x, 4, 5)

#Backwards
str_sub(x, -5,-2)

#Use str_sub to modify strings
str_sub(x, 1, 1) <- str_to_lower(str_sub(x, 1, 1))

#Sort the strings
x <- c("apple", "eggplant", "banana")
str_sort(x, locale = "en")


#####Matching Patterns and Regular Expressions

#Basic
x <- c("apple", "banana", "pear")
str_view(x, "an")
str_view(x, ".a.") #For any character

#When looking for `.`
str_view(c("abc", "a.c", "bef"), "a\\.c")


###Anchors
x <- c("apple", "banana", "pear")
str_view(x, "^a") #Start with a
str_view(x, "a$") #End with a

x <- c("apple pie", "apple", "apple cake")
str_view(x, "apple")
str_view(x, "^apple$") #Contains only

###Repetition
x <- "1888 is the longest year in Roman numerals: MDCCCLXXXVIII"
str_view(x, "CC")
str_view(x, "C+") #One or more


###Detect matches

x <- c("apple", "banana", "pear")

#Checks if e is found in any string of the vector (returns logical)
str_detect(x, "e")

#Words tha start with t in the vector words
summary(str_detect(words, pattern = "^t"))
sum(str_detect(words, "^t"))

# What proportion of common words end with a vowel?
summary(str_detect(words, "[aeiou]$"))
mean(str_detect(string = words, pattern = "[aeiou]$"))


# Find all words containing at least one vowel, and negate
no_vowels_1 <- !str_detect(words, "[aeiou]")
# Find all words consisting only of consonants (non-vowels)
no_vowels_2 <- str_detect(words, "^[^aeiou]+$")
identical(no_vowels_1, no_vowels_2)  #True


#str_subset() get the value of the index position rather than the logical values
words[str_detect(words, "x$")]
str_subset(words, "x$")

#On a data frame
df <- tibble(
  word = words,
  i = seq_along(word))

#Gets the words that end with x
df %>%
  filter(str_detect(words, "x$"))

#Tells you the count of the matches on each value
x <- c("apple", "banana", "pear")
str_count(x, "a")

# On average, how many vowels per word?
mean(str_count(string = words, patter = "[aeiou]"))

#Counts vowels and consonants on each word
df %>%
  mutate(vowels = str_count(string = words, pattern = "[aeiou]"),
         consonants = str_count(string = words, pattern = "[^aeiou]"))

#No overlapping
str_count(string = "abababa", pattern = "aba")
str_view_all("abababa", "aba")  


stringr::sentences
length(sentences)

#Create a list of colors anc place them in a single list
colors <- c("red", "orange", "yellow", "green", "blue", "purple")
color_match <- str_c(colors, collapse = "|")
color_match


has_color <- str_subset(sentences, color_match)
has_color
matches <- str_extract(has_color, color_match) #Extracts the first match
matches

more <- sentences[str_count(sentences, color_match) > 1]
more
str_view_all(more, color_match)
str_extract(more, color_match)
#To get more than the first match
str_extract_all(more, color_match)
str_extract_all(more, color_match, simplify = TRUE) #For a matrix format


####Grouped Matches

noun <- "(a|the) ([^ ]+)"

has_noun <- sentences %>%
  str_subset(noun) %>%
  head(10)

has_noun

has_noun %>%
  str_extract(noun)

#Gives each individual component
has_noun %>%
  str_match(noun)

#Using tibble table
tibble(sentence = sentences) %>%
  tidyr::extract(
    sentence, c("article", "noun"), "(a|the) ([^ ]+)",
    remove = FALSE)

###Replacing Matches

x <- c("apple", "pear", "banana")
str_replace(x, "[aeiou]", "-") #Replaces first voew with `-`

#For all voews
str_replace_all(x, "[aeiou]", "-")

#Replace numbers with words for all vectors
x <- c("1 house", "2 cars", "3 people")
str_replace(string  = x, pattern = c("1", "2", "3"),
            replacement = c("one", "two", "three"))
#Or
str_replace_all(x, c("1" = "one", "2" = "two", "3" = "three"))


###Splitting

sentences %>%
  head(5) %>%
  str_split(" ")

#Use simplify for a matrix representation
sentences %>%
  head(5) %>%
  str_split(" ", simplify = TRUE)

fields <- c("Name: Hadley", "Country: NZ", "Age: 35")
fields %>% str_split(": ", n = 2, simplify = TRUE)

#You can match words, character, sentence from a string using boundary
x <- "This is a sentence. This is another sentence."
str_view_all(x, boundary("word"))

#Ways to separate words from a sentence
str_split(x, " ")[[1]]
str_split(x, boundary("word"))[[1]]

str_locate_all(x, "is")


####Other types of Pattern

# The regular call:
str_view(fruit, "nana")
# Is shorthand for
str_view(fruit, regex("nana"))


bananas <- c("banana", "Banana", "BANANA")
str_view(bananas, "banana")
#Ignores the case sensitive banana
str_view(bananas, regex("banana", ignore_case = TRUE))

x <- "Line 1\nLine 2\nLine 3"
str_extract_all(x, "^Line")[[1]]
#Goes for all the lines
str_extract_all(x, regex("^Line", multiline = TRUE))[[1]]

#Same human character, but defined different
a1 <- "\u00e1"
a2 <- "a\u0301"
c(a1, a2)

#False bc they are defined different
str_detect(a1, a2)
#Checks by the output character and not the defined value
str_detect(a1,coll(a2))

i <- c("I", "İ", "i", "ı")
i
#Ignores upper and lower case
str_subset(string = i , coll("i", ignore_case = T))

#Examples
x <- "This is a sentence"
str_extract_all(string = x, pattern = boundary("word"))


####Other Uses of Regular Expressions 

#Finds functions that ahve that string
apropos("replace")

#Finds files (In here end with .Rmd as the file extension)
head(dir(pattern = "\\.Rmd$"))

###Extra
sentence <- "A very long string lmao"
new <- str_split(string = sentence , pattern = " ")
new[[1]][1]
