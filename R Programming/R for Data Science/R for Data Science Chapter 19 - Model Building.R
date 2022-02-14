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



############### Chapter 19 Model Building

library(tidyverse)
library(modelr)

options(na.action = na.warn)

library(nycflights13)
library(lubridate)

#We can see how cut, color and clarity affect the price of a diamond
ggplot(diamonds, aes(cut, price)) + geom_boxplot()
ggplot(diamonds, aes(color, price)) + geom_boxplot()
ggplot(diamonds, aes(clarity, price)) + geom_boxplot()


#Plot the carat vs price with geom_hex()
ggplot(diamonds, aes(carat, price)) +
  geom_hex(bins = 50)

#Get data whose carat is less than or equal to 2.5 and apply a log function to x & y
diamonds2 <- diamonds %>%
  filter(carat <= 2.5) %>%
  mutate(lprice = log2(price), lcarat = log2(carat))
diamonds2

#Plot filtered data with a log function applied to the carat and price values
ggplot(diamonds2, aes(lcarat, lprice))+
  geom_hex(bins = 50)

#Linear regression
mod_diamond <- lm(lprice ~ lcarat, data = diamonds2)
mod_diamond
summary(mod_diamond)

grid <- diamonds2 %>%
  data_grid(carat = seq_range(carat, 20)) %>%
  mutate(lcarat = log2(carat)) %>%
  add_predictions(mod_diamond, "lprice") %>% #Does a prediction 
  mutate(price = 2 ^ lprice)
grid

#Print with the prediction line
ggplot(diamonds2, aes(carat, price)) +
  geom_hex(bins = 50) +
  geom_line(data = grid, color = "red", size = 1)

#Add residuals 
diamonds2 <- diamonds2 %>%
  add_residuals(mod_diamond, "lresid")

#Plot the carat size vs the residual 
ggplot(diamonds2, aes(lcarat, lresid)) +
  geom_hex(bins = 50)

#Residual vs cut, color and clarity
ggplot(diamonds2, aes(cut, lresid)) + geom_boxplot()
ggplot(diamonds2, aes(color, lresid)) + geom_boxplot()
ggplot(diamonds2, aes(clarity, lresid)) + geom_boxplot()


###A More Complicated Model
mod_diamond2 <- lm(lprice ~ lcarat + color + cut + clarity, data = diamonds2)
mod_diamond2
summary(mod_diamond2)

grid <- diamonds2 %>%
  data_grid(cut, .model = mod_diamond2) %>% #Sort them by cut
  add_predictions(mod_diamond2)
grid

#Similar to predict()
add_predictions(data = diamonds2[c("cut", "lcarat", "color", "clarity")], model = mod_diamond2)

#Plot the preditions for each cut type
ggplot(grid, aes(cut, pred))+
  geom_point()

#Adds the residuals from the second linear model
diamonds2 <- diamonds2 %>%
  add_residuals(mod_diamond2, "lresid2")
print(diamonds2, width = Inf)

#Plot based on log linear model residual vs log carat
ggplot(diamonds2, aes(lcarat, lresid2)) +
  geom_hex(bins = 50)

#Wragling
diamonds2 %>%
  filter(abs(lresid2) > 1) %>%
  add_predictions(mod_diamond2) %>%
  mutate(pred = round(2 ^ pred)) %>%
  select(price, pred, carat:table, x:z) %>%
  arrange(price)


###What affects the number of daily flights

#Gather by numebr per date
daily <- flights %>%
  mutate(date = make_date(year, month, day)) %>%
  group_by(date) %>%
  summarize(n = n())
daily

#Plot the number of fligthts per day
ggplot(daily, aes(date, n))+
  geom_line()

#Per week
daily <- daily %>%
  mutate(wday = wday(date, label = TRUE))

#Flights per day in a week
ggplot(daily, aes(wday, n))+
  geom_boxplot()

#Model
mod <- lm(n ~ wday, data = daily)

#Grid Prediction
grid <- daily %>%
  data_grid(wday) %>% #Sort by wday the prediction
  add_predictions(mod, "n")
grid

#Plot with the prediction
ggplot(daily, aes(wday, n))+
  geom_boxplot() +
  geom_point(data =grid, color = "red", size = 4)

#Residuals
daily <- daily %>%
  add_residuals(mod)
daily

#We can see the deviation from the expected number of fligths in a give day
ggplot(daily, aes(date, resid))+
  geom_ref_line(h = 0)+ #White line at position 0 
  geom_line()

#Now to see the color for each day of the week
ggplot(daily, aes(date, resid, color = wday))+
  geom_ref_line(h = 0)+ #White line at position 0 
  geom_line()
#For saturday there are more fligths than normal on summer and less than in fall

#Some days with far fewers flights than expected
daily %>%
  filter(resid < -100)

#Hilight the smoothness
daily %>%
  ggplot(aes(date, resid)) +
  geom_ref_line(h = 0) +
  geom_line(color = "grey50") +
  geom_smooth(se = FALSE, span = 0.20)

#To accurately predict our fligths on Saturday we see Satuday flights
daily %>%
  filter(wday == "Sat") %>%
  ggplot(aes(date, n)) +
  geom_point() +
  geom_line() +
  scale_x_date( NULL, date_breaks = "1 month", date_labels = "%b")


#We create a term variable to capture the school terms
term <- function(date) {
  cut(date,
      breaks = ymd(20130101, 20130605, 20130825, 20140101), #The four intervals
      labels = c("spring", "summer", "fall")) #The three terms in between intervals
}

#We add the terms to the data based on the date range
daily <- daily %>%
  mutate(term = term(date))
daily

#Plot flights numbers on Saturday on each date with the color on each term
daily %>%
  filter(wday == "Sat") %>%
  ggplot(aes(date, n, color = term)) + 
  geom_point(alpha = 1/3) +
  geom_line() +
  scale_x_date(NULL, date_breaks = "1 month", date_labels = "%b") #Scale by month

#Box plot the terms with the number of flights on each day
daily %>%
  ggplot(aes(wday, n, color = term)) +
  geom_boxplot()

#Linear model with and without the term added 
mod1 <- lm(n ~ wday, data = daily)
mod2 <- lm(n ~ wday * term, data = daily)
summary(mod1)
summary(mod2)

#Plot the residuals for both methods
daily %>%
  gather_residuals(without_term = mod1, with_term = mod2) %>%
  ggplot(aes(date, resid, color = model)) +
  geom_line(alpha = 0.75)

#Same but with the models split  
daily %>%
  gather_residuals(without_term = mod1, with_term = mod2) %>%
  ggplot(aes(date, resid, color = model)) +
  geom_line(alpha = 0.75)+
  facet_wrap(~ model)

#Gather the predictions
grid <- daily %>%
  data_grid(wday, term) %>% #Data grid on wday and term
  add_predictions(mod2, "n") #Add predicted number of flights as n
grid

#Plot the box plot with pred flights num per day per term
ggplot(daily, aes(wday, n)) +
  geom_boxplot() +
  geom_point(data = grid, color = "red") +
  facet_wrap(~ term)

#Robust linear regression
mod3 <- MASS::rlm(n ~ wday * term, data = daily)

#Plot the residuals for the new linear reg model better than mod1 & mod2
daily %>%
  add_residuals(mod3, "resid") %>%
  ggplot(aes(date, resid)) +
  geom_hline(yintercept = 0, size = 2, color = "white") +
  geom_line()


###Computed Variables

#Bundle the creation of variables into a function
compute_vars <- function(data) {
  data %>%
    mutate(term = term(date), wday = wday(date, label = TRUE))
}

#Or apply transformations directly at the models
wday2 <- function(x) wday(x, label = TRUE)
mod3 <- lm(n ~ wday2(date) * term(date), data = daily) 
summary(mod3)


###Time of Year Approach

library(splines) #For ns()

#Robust linear regression
mod <- MASS::rlm(n ~ wday * ns(date, 5), data = daily) #Add natural spline
#ns() generates a basis matrix for representing family of piecewise-cubic splines
#Here each date is produce 5 times ???

#Plot the pred by date and color by day of week
daily %>%
  data_grid(wday, date = seq_range(date, n = 13)) %>%
  add_predictions(mod) %>%
  ggplot(aes(date, pred, color = wday)) +
  geom_line() +
  geom_point()





