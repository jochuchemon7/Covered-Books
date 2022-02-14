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



############### Chapter 18 Model Basics with modelr

library(tidyverse)
library(modelr)
options(na.action = na.warn)

#You can see an upper trend
ggplot(data = sim1, mapping = aes(x,y))+
  geom_point(position = "jitter")


#Making data
models <- tibble(a1 = runif(250, -20, 40), a2 = runif(250, -5, 5))
models

#250 models and not all good, going over different places
ggplot(sim1, aes(x, y)) +
  geom_abline(aes(intercept = a1, slope = a2), data = models, alpha = 1/4) +
  geom_point()

#Copmute equation 7 + x * 1.5 x being x values from sim1
model1 <- function(a, data) {
  a[1] + data$x * a[2]
}
model1(c(7, 1.5), sim1)

#Root mean square deviation for all real points and the predicted linear model
measure_distance <- function(mod, data) {
  diff <- data$y - model1(mod, data)
  sqrt(mean(diff ^ 2))
}
measure_distance(c(7, 1.5), sim1)


#Function for multiple models
sim1_dist <- function(a1, a2) {
  measure_distance(c(a1, a2), sim1)
}

#We compute the dist using random uniform numbers as the models
models <- models %>%
  mutate(dist = purrr::map2_dbl(a1, a2, sim1_dist)) 
models


#Plost the 10 best models
ggplot(sim1, aes(x, y)) +
  geom_point(size = 2, color = "grey30") +
  geom_abline(aes(intercept = a1, slope = a2, color = -dist),
              data = filter(models, rank(dist) <= 10))
#Same as above
ggplot(sim1, aes(x,y)) +
  geom_point(color = "grey30", size =2)+
  geom_abline(data = tail(dplyr::arrange(models, desc(dist)),10),
              mapping = aes(intercept = a1, slope = a2, color = -dist))

#We circle the best models in tearms of smalles distance and x,y for a1 & a2
ggplot(models, aes(a1, a2)) +
  geom_point(data = filter(models, rank(dist) <= 10), size = 4, color = "red") +
  geom_point(aes(colour = -dist))
#Same as above
ggplot(models, aes(a1, a2)) +
  geom_point(data = tail(dplyr::arrange(models, desc(dist))), size = 4, color = "red")+
  geom_point(aes(color = -dist)) 


#Make more models with combinations of 
grid <- expand.grid(a1 = seq(-5, 20, length = 25), a2 = seq(1, 3, length = 25)) %>%
  mutate(dist = purrr::map2_dbl(a1, a2, sim1_dist))
grid

#PLot the models and color in red the to 10 best ones
grid %>%
  ggplot(aes(a1, a2)) +
  geom_point(data = filter(grid, rank(dist) <= 10), size = 4, colour = "red") +
  geom_point(aes(color = -dist))
#Same as above
ggplot(data = grid, aes(a1,a2))+
  geom_point(data = tail(dplyr::arrange(grid, desc(dist)),10),size = 4, color = "red")+
  geom_point(aes(color = -dist)) 

#Plot the models from grid (^) and lay them over the plots of sim1
ggplot(sim1, aes(x, y)) +
  geom_point(size = 2, color = "grey30") +
  geom_abline(aes(intercept = a1, slope = a2, color = -dist),
              data = filter(grid, rank(dist) <= 10))

#Get the best, inital vals are c(0,0)
best <- optim(c(0, 0), measure_distance, data = sim1)
best$par

#Plot the best model from the data it self
ggplot(sim1, aes(x, y)) +
  geom_point(size = 2, color = "grey30") +
  geom_abline(intercept = best$par[1], slope = best$par[2])

#linear model. Similar results than with optim()
sim1_mod <- lm(y ~ x, data = sim1)
sim1_mod


###Visualizating Models

##Predictions

#data_grid the sim1 data
grid <- sim1 %>%
  data_grid(x)
grid

#add_predictions() uses the df and the lm of sim1_mod to do predictions 
#and add a col with the results
grid <- grid %>%
  add_predictions(sim1_mod) 
grid

#Plot the results
ggplot(sim1, aes(x)) +
  geom_point(aes(y = y)) +
  geom_line(aes(y = pred), data = grid, colour = "red", size = 1)

#Same as above
ggplot(sim1, aes(x,y))+
  geom_point() +
  geom_line(data = grid, mapping = aes(y = pred), color = "red", size =1)


##Residuals

#Adds the residual to the data frame
sim1 <- sim1 %>%
  add_residuals(sim1_mod)
sim1

#Plot the count of the residual
ggplot(sim1, aes(resid)) +
  geom_freqpoly(binwidth = 0.5)

#Plot the points of the sim1 and a white line at resid 0
ggplot(sim1, aes(x, resid)) +
  geom_ref_line(h = 0) +
  geom_point()


###Formulas and Model Families

#Data
df <- tribble(
  ~y, ~x1, ~x2,
  4, 2, 5,
  5, 1, 6)
df

#Model Matrix with y and x1 cols 
model_matrix(df, y ~ x1)
model_matrix(df, y ~ x1 - 1) #For no intercept column
model_matrix(df, y ~ x1 + x2) #Using two cols in the equation


###Categorical Variables

#Oof
df <- tribble(
  ~ sex, ~ response,
  "male", 1,
  "female", 2,
  "male", 1)
df
model_matrix(df, response ~ sex)

#Plots sim2 x has 4 categorical characters
ggplot(sim2) +
  geom_point(aes(x, y))
sim2

#Model
mod2 <- lm(y ~ x, data = sim2)
mod2
summary(mod2)

#Predict on each category
grid <- sim2 %>%
  data_grid(x) %>%
  add_predictions(mod2)
grid

#Plots the prediction 
ggplot(sim2, aes(x)) +
  geom_point(aes(y = y)) +
  geom_point( data = grid, aes(y = pred),color = "red",size = 4)

#Error you cant make predictions using the pipe character
tibble(x = "e") %>%
  add_predictions(mod2)

###Interactions

#Data
sim3
#Combining continous and categorical data
ggplot(sim3, aes(x1, y)) +
  geom_point(aes(color = x2))


#Two models one multiplies and the other adds
mod1 <- lm(y ~ x1 + x2, data = sim3)
mod2 <- lm(y ~ x1 * x2, data = sim3)

summary(mod1)
summary(mod2)

#Combine both models
grid <- sim3 %>%
  data_grid(x1, x2) %>%
  gather_predictions(mod1, mod2)
grid

#Plot both models side by side
ggplot(data = sim3, aes(x = x1, y = y, color = x2))+
  geom_point() +
  geom_line(data = grid, aes(y = pred))+
  facet_wrap(~ model)

#Places the reisudal based on each model for all the rows
Sim3 <- sim3 %>%
  gather_residuals(mod1, mod2)

#Plotting the residuals on both models for each x2 category
ggplot(Sim3, aes(x1, resid, color = x2)) +
  geom_point() +
  facet_grid(model ~ x2)


##Take Two

#Models for sim4
mod1 <- lm(y ~ x1 + x2, data = sim4)
mod2 <- lm(y ~ x1 * x2, data = sim4)

#Predictions 
grid <- sim4 %>%
  data_grid(x1 = seq_range(x1, 5),x2 = seq_range(x2, 5)) %>% #Like expand_grid
  gather_predictions(mod1, mod2)
grid

#Gives 5 equality spread apart values in range
seq_range(c(0.0123, 0.923423), n = 5)

#For rounded numebrs
seq_range(c(0.0123, 0.923423), n = 5, pretty = TRUE)

#100 values almost random 
x1 <- rcauchy(100)
x1

#Using trim
seq_range(x1, n = 5)
seq_range(x1, n = 5, trim = 0.10) #Trims that rate of tail values
seq_range(x1, n = 5, trim = 0.25)
seq_range(x1, n = 5, trim = 0.50)

#Using expand
seq_range(x1, n = 5)
seq_range(x1, n = 5, expand = 0.10) #Expands that rate of tail values
seq_range(x1, n = 5, expand = 0.25)
seq_range(x1, n = 5, expand = 0.50)

#Visualize the model using geom_tile() like a 3D plane
ggplot(grid, aes(x1, x2)) +
  geom_tile(aes(fill = pred)) +
  facet_wrap(~ model)

#Plot x1 by pred and group on x2 compare the two models
ggplot(grid, aes(x1, pred, color = x2, group = x2)) +
  geom_line() +
  facet_wrap(~ model)

#Plot x2 by pred and color/group by x2 
ggplot(grid, aes(x2, pred, color = x1, group = x1)) +
  geom_line() +
  facet_wrap(~ model)


###Transformations

#Data
df <- tribble(
  ~y, ~x,
  1, 1,
  2, 2,
  3, 3)
df

#You can see what equation is lm() fitting
model_matrix(df, y ~ x^2 + x)
#Same as above
model_matrix(df, y ~ I(x^2) + x)

#Check what equation with poly 
model_matrix(df, y ~ poly(x, 2)) #Copmutes otrhogonal polynomials


library(splines)
model_matrix(df, y ~ ns(x, 2))


sim5 <- tibble(x = seq(0, 3.5 * pi, length = 50), 
               y = 4 * sin(x) + rnorm(length(x)))
sim5 

ggplot(sim5, aes(x, y)) +
  geom_point()


mod1 <- lm(y ~ ns(x, 1), data = sim5)
mod2 <- lm(y ~ ns(x, 2), data = sim5)
mod3 <- lm(y ~ ns(x, 3), data = sim5)
mod4 <- lm(y ~ ns(x, 4), data = sim5)
mod5 <- lm(y ~ ns(x, 5), data = sim5)

grid <- sim5 %>%
  data_grid(x = seq_range(x, n = 50, expand = 0.1)) %>%
  gather_predictions(mod1, mod2, mod3, mod4, mod5, .pred = "y")

ggplot(sim5, aes(x, y)) +
  geom_point() +
  geom_line(data = grid, color = "red") +
  facet_wrap(~ model)

