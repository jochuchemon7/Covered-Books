library(caret)
library(dplyr)
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
library(MASS)
library(ROCR)
library(grid)
library(ggpubr)

#####################Chapter 8 Unsupervised Methods

#Use "curl LJO " to download raw github data
getwd()
setwd("Desktop/R Work/Learning/Practical Data Science With R/Data/")
protein <- read.table("protein.txt", header = T, sep = "\t")
setwd("/Users/Josue")

summary(protein)
#There is no unit of measure for the protein
head(protein)
# := creates a column on a df ei (df[,name := values])

###----Rescaling the data

#Colnames exepct country
vars.to.use <- colnames(protein)[-1]

#Scales the function all except with the country col 
#(So the data attributes have the same type of unit measurement) mean = 0, sd = 1
pmatrix <- scale(protein[,vars.to.use])
#Mean values of all columns 
pcenter <- attr(pmatrix, "scaled:center")
#Standard deviation values of all columns
pscale <- attr(pmatrix, "scaled:scale")
#So you can unscale later

###---Hierarchical clustering

#Computes the distance matrix (Distances between the rows of a df, for each value)
d <- dist(pmatrix, method="euclidean")
#Creates the hierarchical cluster with the distances and the method as 'ward'
#Wrds starts at each point as an individual cluster
pfit <- hclust(d, method="ward")
#Plots the hierarchical model with the x-axis being the countries labeled
plot(pfit, labels=protein$Country) #We deduce how many clusters it suggests

#Rectangles exacly 5 clusters of country list (better for vizualization)
rect.hclust(pfit, k=5)

#Cuts the tree into 5 clusters from the hcluster tree (by numerica category)
groups <- cutree(pfit, k=5)
#Numbers given in the order of entries from the labels ei (protein$Country)

#WSS -> to minimize the total within sum of squares of the cluster

#Function to gather the countries and info on the cluster number 
print_clusters <- function(labels, k) {
  for(i in 1:k) {
    print(paste("cluster", i))
    #Given the cluster number 
    print(protein[labels==i,c("Country","RedMeat","Fish","Fr.Veg")])
  }
}

#Prints the function that gahers the countries by cluster number classifier
print_clusters(groups, 5)

#Projecting the clusters on the first two principal components
library(ggplot2)
#Calulate the principal components of pmatrix
# You can also use princomp(pmatrix) and then $loadings 
#If princomp(pmatrix) -> summary(pca) use components with the highest 'prop of var'
princ <- prcomp(pmatrix)

#To view the attributes change on the different PC
princ

#To View the Proportion of Variance (description of data percent)
summary(princ)

nComp <- 2 #Index for the second principàl component
#Makes prediction 
project <- predict(princ, newdata=pmatrix)[,1:nComp]

#A Data frame with the two predicted PC and the groups as cluster and countries
project.plus <- cbind(as.data.frame(project), cluster=as.factor(groups),
                      country=protein$Country)
#Plot the PCA 1 and PCA 2 and mark the clusters and label the text to the countries
ggplot(project.plus, aes(x=PC1, y=PC2)) +
  geom_point(aes(color=cluster)) +
  geom_text(aes(label=country), hjust=0, vjust=1)

#Using autoplot for the pca plotting
library(ggfortify)
autoplot(princ, data = protein, colour = 'Country')

#To check the variances among the PC's
plot(princ, type ='l')


######My Example (Following the book)
################################################################################
pca <- prcomp(pmatrix)
pca

#Predict using only the first two principal components
pred <- predict(pca, newdata = pmatrix)[,1:2]

#Combine the two pc and the cluster/group and countries into one df
pred.plus <- cbind(data.frame(pred), cluster = as.factor(groups), 
                   countries = protein$Country)

#Plot them
ggplot(pred.plus, aes(x = PC1, y = PC2))+
  geom_point(aes(shape = cluster))+
  geom_text(aes(label = countries), hjust = 0 , vjust =1)
################################################################################

###-----Running clusterboot() on the protein data Using Hierachical Clustering

library(fpc)
kbest.p<-5

#Run clustering by resampling 
cboot.hclust <- clusterboot(pmatrix,clustermethod=hclustCBI,
                            method="ward", k=kbest.p)

#Summary on the cboot.hclust$result
summary(cboot.hclust$result)

#Store the partition from the result on the cboot.hclust 
# (The new Groups from previous hclust)
groups<-cboot.hclust$result$partition

#Print the new distribution of clusters by 5 clusters
print_clusters(groups, kbest.p)

#Vector cluster stability (Below .6 are unstable) .6-.75 somewhat. > .85 highly
# Stable 
cboot.hclust$bootmean

#The counts of how many times each cluster was dissolved.
#By default clusterboot() runs 100 bootstrap iterations
cboot.hclust$bootbrd

#Calculating total within sum of squares
sqr_edist <- function(x, y) {
  sum((x-y)^2)
}

#Find the sum of all the sum of squares on all
wss.cluster <- function(clustermat) {
  c0 <- apply(clustermat, 2, FUN=mean)
  sum(apply(clustermat, 1, FUN=function(row){sqr_edist(row,c0)}))
}

wss.total <- function(dmatrix, labels) {
  wsstot <- 0
  k <- length(unique(labels))
  for(i in 1:k)
    wsstot <- wsstot + wss.cluster(subset(dmatrix, labels==i))
  wsstot
}

#####------The Calinski-Harabasz index (CH criterion)

#Calinski-Harabasz Index
calinhara(x,clustering,cn=max(clustering))

#Convenience function to calculate the total sum of squares.
totss <- function(dmatrix) {
  grandmean <- apply(dmatrix, 2, FUN=mean)
  sum(apply(dmatrix, 1, FUN=function(row){sqr_edist(row, grandmean)}))
}

#Calculates the CH index 
ch_criterion <- function(dmatrix, kmax, method="kmeans") {
  if(!(method %in% c("kmeans", "hclust"))) { #Checks for cluster method existance
    stop("method must be one of c('kmeans', 'hclust')")
  }
  
  npts <- dim(dmatrix)[1] # number of rows.
  totss <- totss(dmatrix)
  wss <- numeric(kmax)
  crit <- numeric(kmax)
  wss[1] <- (npts-1)*sum(apply(dmatrix, 2, var))
  for(k in 2:kmax) {
    if(method=="kmeans") {
      clustering<-kmeans(dmatrix, k, nstart=10, iter.max=100)
      wss[k] <- clustering$tot.withinss
    }else { # hclust
      d <- dist(dmatrix, method="euclidean")
      pfit <- hclust(d, method="ward")
      labels <- cutree(pfit, k=k)
      wss[k] <- wss.total(dmatrix, labels)
    }
  }
  bss <- totss - wss
  crit.num <- bss/(0:(kmax-1))
  crit.denom <- wss/(npts - 1:kmax)
  list(crit = crit.num/crit.denom, wss = wss, totss = totss)
}
  

#Evaluating clusterings with different numbers of clusters

library(reshape2)

#Calls the ch criterion fun
clustcrit <- ch_criterion(pmatrix, 10, method="hclust")
clustcrit

#CH index and wss for each cluster index
critframe <- data.frame(k=1:10, ch=scale(clustcrit$crit), 
                        wss=scale(clustcrit$wss))

#melt would take the variable and pair it with its value (We indicate the k values
# as our id.variables ei. (grouped by))
critframe <- melt(critframe, id.vars=c("k"), variable.name="measure",
                  value.name="score")

#Plot it
ggplot(critframe, aes(x=k, y=score, color=measure)) +
  geom_point(aes(shape=measure)) + 
  geom_line(aes(linetype=measure)) +
  scale_x_continuous(breaks=1:10, labels=1:10)


####**********--Using Kmeans on the protein data

#Use kmeans() function (pmatrix is still scaled)
pclusters <- kmeans(pmatrix, kbest.p, nstart=100, iter.max=100)
              #100 random start sets are choosen AND 100 max num of iterations
summary(pclusters)#Info on the size of info available

#rows are the centroids of the clusters (based on scale protein data)
pclusters$center

#The new group cluster 
groups <- pclusters$cluster
print_clusters(groups, kbest.p) #Print Info on the countries in each cluster

####--------Plotting cluster criteria

#Calls the kmeans but initializes the kmeans algo several times with random pts
clustering.ch <- kmeansruns(pmatrix, krange=1:10, criterion="ch")
                  #num of clusters AND calinkski criterion
clustering.ch$bestk #Gets the best number k clusters 

#Same but using the asw criterion
clustering.asw <- kmeansruns(pmatrix, krange=1:10, criterion="asw")
clustering.asw$bestk #Gets the best number of k clusters

#The criterion values of ch kmeans
clustering.ch$crit
clustcrit$crit #ch criterion value using the hclust/ hierachical

oldcritframe <- critframe #Save previous frame
#New critframe using the kmeans with criterions asw and ch
critframe <- data.frame(k=1:10, ch=scale(clustering.ch$crit),
                          asw=scale(clustering.asw$crit))
#Melt the data
critframe <- melt(critframe, id.vars=c("k"), variable.name="measure",
                    value.name="score")
#Plot score based on the measure
ggplot(critframe, aes(x=k, y=score, color=measure)) +
  geom_point(aes(shape=measure)) + geom_line(aes(linetype=measure)) +
  scale_x_continuous(breaks=1:10, labels=1:10)


####------Running clusterboot() with k-means

#Number of clusters
kbest.p<-5

#Run kmean with 100 run and iterations with 5 k's and method kmeansCBI
cboot<-clusterboot(pmatrix, clustermethod=kmeansCBI, runs=100,iter.max=100,
                   krange=kbest.p, seed=15555)

#Partition gives the cluster number assign to each entry from the data 
groups <- cboot$result$partition
#We print th clusters with info
print_clusters(cboot$result$partition, kbest.p)

#The stability numbers
cboot$bootmean
#The number of times the clusters where 'dissolved'
cboot$bootbrd

####---------A function to assign points to a cluster

assign_cluster <- function(newpt, centers, xcenter=0, xscale=1) {
  #Center and scale the new data point
  xpt <- (newpt - xcenter)/xscale
  #Calculates the square sum of how far the points are form each other (dist)
  dists <- apply(centers, 1, FUN=function(c0){sqr_edist(c0, xpt)})#square euclidian
  which.min(dists) #Returns the cluster number closest to the centroid
}

###------Example of assigning the points to the clusters

#A function to generate n points drawn from a multidimensional Gaussian 
#distribution with centroid mean and standard deviation sd
rnorm.multidim <- function(n, mean, sd, colstr="x") {
  ndim <- length(mean)
  data <- NULL
  for(i in 1:ndim) {
    col <- rnorm(n, mean=mean[[i]], sd=sd[[i]])
    data<-cbind(data, col)
  }
  cnames <- paste(colstr, 1:ndim, sep='')
  colnames(data) <- cnames
  data
}

#Parameter examples
mean1 <- c(1, 1, 1)
sd1 <- c(1, 2, 1)

mean2 <- c(10, -3, 5)
sd2 <- c(2, 1, 2)

mean3 <- c(-5, -5, -5)
sd3 <- c(1.5, 2, 1)

#Create the random normalized multidim data with diff mean set and sd 
clust1 <- rnorm.multidim(100, mean1, sd1)
clust2 <- rnorm.multidim(100, mean2, sd2)
clust3 <- rnorm.multidim(100, mean3, sd3)
#Row combine all the data
toydata <- rbind(clust3, rbind(clust1, clust2))

#Scale the toydata
tmatrix <- scale(toydata)
tcenter <- attr(tmatrix, "scaled:center")
tscale<-attr(tmatrix, "scaled:scale")
kbest.t <- 3
tclusters <- kmeans(tmatrix, kbest.t, nstart=100, iter.max=100)
tclusters

tclusters$size

unscale <- function(scaledpt, centervec, scalevec) {
  scaledpt*scalevec + centervec
}

unscale(tclusters$centers[1,], tcenter, tscale)
mean2

unscale(tclusters$centers[2,], tcenter, tscale)
mean3

unscale(tclusters$centers[3,], tcenter, tscale)
mean1

assign_cluster(rnorm.multidim(1, mean1, sd1),
                 tclusters$centers,
                 tcenter, tscale)
assign_cluster(rnorm.multidim(1, mean2, sd1),
                 tclusters$centers,
                 tcenter, tscale)
assign_cluster(rnorm.multidim(1, mean3, sd1),
               tclusters$centers,
               tcenter, tscale)

