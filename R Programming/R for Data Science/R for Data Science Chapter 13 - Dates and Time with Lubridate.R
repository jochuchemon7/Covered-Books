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



############### Chapter 13 - Dates and Time with Lubridate

library(tidyverse)
library(lubridate)
library(nycflights13)



###Creating Date/Times

today() #Date
now() #Date/Time


###From Strings

#They give the same Date variable type
ymd("2017-01-31")
mdy("January 31st, 2017")
dmy("31-Jan-2017")

#With just int to character
ymd(20170131)

#Full Date/time w/out seconds
ymd_hms("2017-01-31 20:11:59")
mdy_hm("01-21-2020 12:29")

#You can supply the time zone
ymd(20170131, tz = "UTC")

#Individual Components
as.data.frame(head(flights)) %>%
  dplyr::select(year, month, day, hour, minute)

#Create date/time
as.data.frame(head(flights)) %>%
  dplyr::select(year, month, day, hour, minute) %>%
  mutate(created_date = make_datetime(year, month, day, hour, minute))


#Using the flights time we make our own
make_datetime_100 <- function(year, month, day, time) {
  make_datetime(year, month, day, time %/% 100, time %% 100)
}

#Takes the y,m,d and time of hmm and makes it in to a date time
flights_dt <- flights %>%
  filter(!is.na(dep_time), !is.na(arr_time)) %>%
  mutate(dep_time = make_datetime_100(year, month, day, dep_time),
         arr_time = make_datetime_100(year, month, day, arr_time),
         sched_dep_time = make_datetime_100(year, month, day, sched_dep_time),
         sched_arr_time = make_datetime_100(year, month, day, sched_arr_time)) %>%
  dplyr::select(origin, dest, ends_with("delay"), ends_with("time"))

flights_dt
print(flights_dt, width = Inf)

#We plot the count of flights on each dep date
ggplot(flights_dt,aes(dep_time)) +
  geom_freqpoly(binwidth = 86400)

#We plot the flight count on one day; spread in time since the date is just one
flights_dt %>%
  filter(dep_time < ymd(20130102)) %>%
  ggplot(aes(dep_time)) +
  geom_freqpoly(binwidth = 600)

#To see the count of flifghts on each time recorded
flights_dt %>%
  filter(dep_time < ymd(20130102)) %>%
  group_by(dep_time) %>%
  count() %>%
  arrange(desc(n))


###From Other Types

#Ways to gather todays date
today()
as_datetime(today())

now() #Date 
as_date(now()) #To date, drop time

as_datetime(60 * 60 * 10)
as_date(365 * 10 + 2)


###Date Time Components
datetime <- ymd_hms("2016-07-08 12:34:56")
datetime

year(datetime) #Get year number
month(datetime) #Gets month number
day(datetime) #Gets day number (Also ´mday()´ works)
hour(datetime) #Gets hour
minute(datetime) #Gets minute
second(datetime) #Gets second
yday(datetime) #Day of the year
wday(datetime) #Day of the week
mday(datetime) #Day of the month

#Show name of day, month with label = T
month(datetime, label = TRUE) #With the month abrebiation 
wday(datetime, label = TRUE, abbr = FALSE) #With day name

#Plot the count of flights on each day
ggplot(flights_dt, aes(x = wday(dep_time, label = T))) +
  geom_bar()+
  xlab("wday")

#Same as ^
flights_dt %>%
  mutate(wday = wday(dep_time, label = TRUE)) %>%
  ggplot(aes(x = wday)) +
  geom_bar()

#Plots the avg delay for minutes of dep_time away from each hour on dep times
flights_dt %>%
  mutate(minutes = minute(dep_time)) %>%
  group_by(minutes) %>%
  summarize(avg_delay = mean(arr_delay, na.rm = T), n = n()) %>%
  ggplot(aes(x = minutes, y = avg_delay)) +
  geom_line()

#Now avg_delay for minutes away from hour on scheduled dep times
sched_dep <- flights_dt %>%
  mutate(minute = minute(sched_dep_time)) %>%
  group_by(minute) %>%
  summarize(avg_delay = mean(arr_delay, na.rm = TRUE), n = n())
ggplot(sched_dep, aes(minute, avg_delay)) +
  geom_line()

#We see a disparity of avg_delay tim eis higher when the dep time is closer to 
#a whole hour, also bc more departures occur when the hour is close to whole
ggplot(sched_dep, aes(minute, n)) +
  geom_line()


###Rounding

#We can round the time by day, week, month or year so to `group_by`
#Here we count by week
flights_dt %>%
  count(week = floor_date(dep_time, "week")) %>% #Also ceiling_date()
  ggplot(aes(week, n)) +
  geom_line()


###Setting Components

(datetime <- ymd_hms("2016-07-08 12:34:56"))

#Change the year on a POSIXct
year(datetime) <- 2020
datetime
#Change the month on a POSIXct
month(datetime) <- 01
datetime
#Change the hour on a POSIXct by adding one more
hour(datetime) <- hour(datetime) + 1
datetime

#Also use update for multiple variables
update(datetime, year = 2020, month = 2, mday = 2, hour = 2)

#Over set and it rolls over to the next higest val
ymd("2015-02-01") %>%
  update(mday = 30)

ymd("2020-07-16") %>%
  update(hour = 40)

#Show distribution of flights across the course of the day for every day of the year
flights_dt %>%
  mutate(dep_hour = update(dep_time, yday = 1)) %>%
  ggplot(aes(dep_hour)) +
  geom_freqpoly(binwidth = 300)


###Time Spans

##Durations: represent an exact number of seconds

#Get difference in days
h_age <- today() - ymd(19791014)
h_age

#Duration in seconds using as.duration()
as.duration(h_age)

dseconds(15) #Duration (sec) from 15 seconds
dminutes(10) #Duration (sec) from 10 minutes
dhours(c(12, 24)) #Duration (sec) from 12 and 24 hours
ddays(0:5) #Duration (sec) from 0,1,2,3,4,5 days
dweeks(12) #For 12 weeks
dyears(20) #For 20 years

2 * dyears(1) #You can multiply 

dyears(1) + dweeks(12) + dhours(15) #Or Add the seconds on week, hour, or year based

#Arithmetic
tomorrow <- today() + ddays(1) 
tomorrow
last_year <- today() - dyears(1)
last_year

one_pm <- ymd_hms("2016-03-12 13:00:00", tz = "America/New_York")
one_pm
one_pm + ddays(1) #When adding a day long in seconds an extra hour is added too


##Periods: represent human units like weeks and months.

one_pm
one_pm + days(1) #Use days instead

#Fill the time given and the lower value variables 
seconds(15)
minutes(10)
hours(c(12, 24))
days(7)
months(1:6)
weeks(3)
years(1)

# You can do more arihmetic with days and months
10 * (months(6) + days(1))
days(50) + hours(25) + minutes(2) #Or Just assing the values 

# A leap year
ymd("2016-01-01") + dyears(1)
ymd("2016-01-01") + years(1)

# Daylight Savings Time
one_pm + ddays(1)
one_pm + days(1)

#Some places arrived before they departed
flights_dt %>%
  filter(arr_time < dep_time)

flights_dt <- flights_dt %>%
  mutate(overnight = arr_time < dep_time,
         arr_time = arr_time + days(overnight * 1),
         sched_arr_time = sched_arr_time + days(overnight * 1))

#Laws of physics are not enforced
flights_dt %>%
  filter(overnight,arr_time < dep_time)


###Intervals

years(1) / days(1)
dyears(1) / ddays(365) #Not Fully one

#Get next year date
next_year <- today() + years(1)
next_year

(today() %--% next_year) / ddays(1) #Days appart
(today() %--% next_year) %/% days(1) #Days appart


###Time Zones

#Get your timezome
Sys.timezone()

#Number of timezones
length(OlsonNames())

#Example
head(OlsonNames())

#All represent the same instance in time
(x1 <- ymd_hms("2015-06-01 12:00:00", tz = "America/New_York"))
(x2 <- ymd_hms("2015-06-01 18:00:00", tz = "Europe/Copenhagen"))
(x3 <- ymd_hms("2015-06-02 04:00:00", tz = "Pacific/Auckland"))

#You can verify by substracting 
x1 - x2

#If same instances, tz would be replace by your local one
x4 <- c(x1, x2, x3)
x4

#Change time zones
x4a <- with_tz(x4, tzone = "Australia/Lord_Howe")
x4a

#No time difference
x4a - x4

#Change time zone
x4b <- force_tz(x4, tzone = "Australia/Lord_Howe")
x4b
x4b - x4 #Difference of 14.5










