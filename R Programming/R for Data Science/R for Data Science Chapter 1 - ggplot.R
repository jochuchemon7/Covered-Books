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


################### Chapter 1 ggplot

#mpg data and dysp by highwa

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy))

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy, color = class)) #Or size = class

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy, alpha = class)) #Or shape = class

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy), color = "blue")

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy)) +
  facet_wrap(~ class, nrow = 2)

ggplot(data = mpg) +
  geom_point(mapping = aes(x = displ, y = hwy)) +
  facet_grid(drv ~ cyl)


### Geometric Objects

ggplot(data = mpg) +
  geom_smooth(mapping = aes(x = displ, y = hwy))

ggplot(data = mpg) +
  geom_smooth(mapping = aes(x = displ, y = hwy, linetype = drv))

ggplot(data = mpg) +
  geom_smooth(mapping = aes(x = displ, y = hwy, color = drv)) +
  geom_point(mapping = aes(x = displ, y = hwy, color = drv))

#For global mapping
ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color = drv)) +
  geom_smooth()+
  geom_point()

#Any mapping done in any geom is for that local one and would override it

#Their own subdata and color for their own geom
ggplot(data  = mpg, mapping = aes(x = displ, y = hwy))+
  geom_point(mapping = aes(color = class)) +
  geom_smooth(data = filter(mpg , class == "subcompact"), se = F, color = "blue") +
  geom_smooth(data = filter(mpg , class == "compact"), se = F, color = "red") 


########## Statistical Transformations

ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut))

#Similar to ^
ggplot(data = diamonds) +
  stat_count(mapping = aes(x = cut))

ggplot(data = diamonds)+
  geom_bar(mapping = aes(x = cut))#Using stat = overrides the geom_bar and stat_count

ggplot(data = diamonds) +
  stat_summary(
    mapping = aes(x = cut, y = depth),
    fun.ymin = min,
    fun.ymax = max,
    fun.y = median
  )

ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut, fill = cut))

ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut, fill = clarity))

#Make the bars proportional
ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut, fill =  clarity), position = "fill")

#Count but all permutations for aes(x and fill)
ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut, fill =  clarity), position = "dodge")

#For scatter to avod overplotting and to see clusters we can use jitter for position
ggplot(data = mpg) +
  geom_point(
    mapping = aes(x = displ, y = hwy),
    position = "jitter"
  )

ggplot(data = mpg, mapping = aes(x = cty, y = hwy))+
  geom_point(position = "jitter") #Without jitter it would look weird

######## Cordinate Systems
#par(mfrow = c(1, 2))

#coord_flip() switches the x and y axis
ggplot(data = mpg, mapping = aes(x = class, y = hwy))+
  geom_boxplot()+
  coord_flip()

#New Zeland Map
nz <- map_data("nz")
ggplot(nz, mapping = aes(x = long, y = lat, group = group)) +
  geom_polygon(fill = "white", color = "black") +
  coord_quickmap() #To fit the map


bar <- ggplot(data = diamonds) +
  geom_bar(mapping = aes(x = cut, fill = cut), show.legend = FALSE, width = 1) +
  theme(aspect.ratio = 1) +
  labs(x = NULL, y = NULL)

#Uses the polar coordinates
bar + coord_flip()
bar + coord_polar()



####Extra
library(raster)
mex <- getData(name = "GADM", country = "Mexico", level = 2)

ggplot(mex,aes(x=long,y=lat,group=group))+
  geom_polygon(aes(fill=id),color="grey30") +
  scale_fill_discrete(guide="none")+
  theme_bw()+theme(panel.grid=element_blank())




vietnam  <- getData("GADM",country="Vietnam",level=2)

ggplot(vietnam,aes(x=long,y=lat,group=group))+
  geom_polygon(fill="white",color="grey30") +
  coord_quickmap()





