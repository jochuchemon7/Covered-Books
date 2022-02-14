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



############### Chapter 22 Graphics For Communication with ggplot2

library(tidyverse)

#Plot mpg displ vs hwy
ggplot(mpg, aes(displ, hwy))+
  geom_point(aes(color = class))+
  geom_smooth(se = F)+
  labs(title = paste("Fuel efficiency generally decreases with engine size"))
  
#Plot with title, subtitle and caption
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  labs(title = paste("Fuel efficiency generally decreases with engine size"),
    subtitle = paste("Two seaters (sports cars) are an exception because of their light weight"),
    caption = "Data from fueleconomy.gov")

#Plot with x, y labels and for colors for the color aes
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  labs(x = "Engine displacement (L)",y = "Highway fuel economy (mpg)",
       colour = "Car type")

#Data 
df <- tibble(x = runif(10),
  y = runif(10))
df

#Plot, the x & y labels are equations
ggplot(df, aes(x, y)) +
  geom_point() +
  labs(x = quote(sum(x[i] ^ 2, i == 1, n)),#Use quote()
       y = quote(alpha + beta + frac(delta, theta)))


###Anotations

#row_number() gets the index of the sorted vector/list- It gets the highest hwy 
best_in_class <- mpg %>% ##Provides the labels
  group_by(class) %>%
  filter(row_number(desc(hwy)) == 1)
best_in_class


#Plot with geom_text for text of best hwy model in the data, from data best_in_class
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_text(aes(label = model), data = best_in_class)

#Looks better with geom_label (encapsulates it)
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_label(aes(label = model),data = best_in_class, nudge_y = 2, alpha = 0.5)

#Or to completely avoid overlaps use ggrepel::geom_label_repel()
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_point(size = 3, shape = 1, data = best_in_class) + #Mark the best in class
  ggrepel::geom_label_repel(aes(label = model),
    data = best_in_class)

#Get the class hwy and displ median
class_avg <- mpg %>%
  group_by(class) %>%
  summarize(displ = median(displ),
    hwy = median(hwy))
class_avg

#Plot using the theme lege.positiion = "none"
ggplot(mpg, aes(displ, hwy, color = class)) +
  ggrepel::geom_label_repel(aes(label = class),
                            data = class_avg,
                            size = 6,
                            label.size = 0,
                            segment.color = NA) +
  geom_point() +
  theme(legend.position = "none")

#Gather the labels
label <- mpg %>%
  summarize(displ = max(displ), hwy = max(hwy),
    label = paste("Increasing engine size is \nrelated to decreasing fuel economy."))
label

#Plot with the labels on the top right corner
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_text(aes(label = label), data = label, vjust = "top",hjust = "right")


#New Label
label <- tibble(displ = Inf, hwy = Inf,
  label = paste("Increasing engine size is \nrelated to",
    "decreasing fuel economy."))
label

#With Inf displ & hwy the label goes at the very corner 
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_text(aes(label = label), data = label, vjust = "top", hjust = "right")


###Scales

#Regular Plot
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class))

#The behind the scenes of ^
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  scale_x_continuous() +
  scale_y_continuous() +
  scale_color_discrete()


###Axis Ticks and Legend Keys

#Plot y axis from marking at 15 to 40 in sequences of 5
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  scale_y_continuous(breaks = seq(15, 40, by = 5)) 

#Plot withouth the numeric label on the axis
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  scale_x_continuous(labels = NULL) +
  scale_y_continuous(labels = NULL)

#Plot removes label for x axis
presidential %>%
  mutate(id = 33 + row_number()) %>%
  ggplot(aes(start, id)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_x_date(NULL, breaks = presidential$start, date_labels = "'%y") 


###Legend Layout

#You can choose the location of the legend
base <- ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class))
base
base + theme(legend.position = "left")
base + theme(legend.position = "top")
base + theme(legend.position = "bottom")
base + theme(legend.position = "right")

#Plot the with legend.position = bottom
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 1, override.aes = list(size = 4)))



###Replacing a scale

#Plot using geom_bin2d()
ggplot(diamonds, aes(carat, price)) +
  geom_bin2d()

#Same but with the logarithmic transformation the label values change to the log transformation
ggplot(diamonds, aes(log10(carat), log10(price))) +
  geom_bin2d()

#Same ^ but keep the original label values
ggplot(diamonds, aes(carat, price)) +
  geom_bin2d() +
  scale_x_log10() + 
  scale_y_log10()

#Another plot
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = drv))

#Using scale_color_brewer() for those color blinded
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = drv)) +
  scale_color_brewer(palette = "Set1")

#Add Shape so it can be read in black or white
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = drv, shape = drv)) +
  scale_color_brewer(palette = "Set1")


#Plot and color the party and bc color = party, you can just indicate the color to the values
presidential %>%
  mutate(id = 33 + row_number()) %>%
  ggplot(aes(start, id, color = party)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_colour_manual(values = c(Republican = "red", Democratic = "blue"))

#Data
df <- tibble(x = rnorm(10000),
  y = rnorm(10000))
df

#Plotting using coord_fixed() 
ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed()

#Plot you can change the color of the heat map
ggplot(df, aes(x, y)) +
  geom_hex() +
  viridis::scale_fill_viridis() +
  coord_fixed()


###Zooming

#Plots with defined set of x and y value limits to be shown 
ggplot(mpg, mapping = aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth() +
  coord_cartesian(xlim = c(5, 7), ylim = c(10, 30))

#Same as ^ but using filter in terms of shown the data point
mpg %>%
  filter(displ >= 5, displ <= 7, hwy >= 10, hwy <= 30) %>%
  ggplot(aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth()


suv <- mpg %>% 
  filter(class == "suv")
suv

compact <- mpg %>%
  filter(class == "compact")
compact

#Select only the desired class to be ploted from the passed data
ggplot(suv, aes(displ, hwy, color = drv)) +
  geom_point()
ggplot(compact, aes(displ, hwy, color = drv)) +
  geom_point()

#Assigning the ranges of displ, hwy and drv to the scales that are going to be used
x_scale <- scale_x_continuous(limits = range(mpg$displ))
y_scale <- scale_y_continuous(limits = range(mpg$hwy))
col_scale <- scale_color_discrete(limits = unique(mpg$drv))

#Plot by using the scale functions passed for suv data
ggplot(suv, aes(displ, hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale

#Same but for compact data
ggplot(compact, aes(displ, hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale


###Themes

#Plot using a black and white theme
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme_bw()

#Using dark theme
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme_dark()


###Saving the Plots

#Regular Plot
ggplot(mpg, aes(displ, hwy)) + geom_point()

#Saves the most recent plot
ggsave("my-plot.pdf")


###Google maps like it
library(leaflet)

leaflet() %>% #World map
  setView(174.764, -36.877, zoom = 16) %>% #Gives the coordinates
  addTiles() %>%
  addMarkers(174.764, -36.877, popup = "Maungawhau") #Adds the big marker



###Shiny

library(shiny)
textInput("name", "What is your name?")
numericInput("age", "How old are you?", NA, min = 0, max = 150)





