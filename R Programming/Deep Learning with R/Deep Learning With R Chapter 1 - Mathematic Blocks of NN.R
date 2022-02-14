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
library(keras)
library(sparklyr)
#library(imager) Won't work
library(png) 
library(RCurl)

#####################Chapter 1 - Mathematics of NN

#use_condaenv("r-tensorflow")


#tensorflow::install_tensorflow()

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


#View the greyscale images
for (i in 1:10) {
  image(test_images[i,1:28,1:28], col = gray((0:255)/255))
}


str(train_images)
str(train_labels)
str(test_images)
str(test_labels)


#The fully connected network
Network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")

#The compilation step
Network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


#Preparing the Image Data by 3D -> 2D and normalizing the greyscale
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

#Preparing the Labels - from dim(10000,1) to dim(10000, 10) essentially a dummyVars 
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


#Training the NN
Network %>%
  fit(train_images, train_labels, epochs = 5, batch_size = 128)


#Evaluate the Network with the test images 
metrics <- Network %>% 
  evaluate(test_images, test_labels)

metrics

#View the greyscale images (Just for reference)
for (i in 1:10) {
  image(mnist$test$x[i,1:28,1:28], col = gray((0:255)/255))
}

#Generate Predictions
Network %>% 
  predict_classes(test_images[1:10,])
#Or by
predict_classes(object = Network, x = test_images[1:10,])


#View the number of axes in the data
length(dim(mnist$train$x))
#Shape
dim(mnist$train$x)
#Data Type
typeof(mnist$train$x)

#Show the 5th digit
digit <- mnist$train$x[5,,]
plot(as.raster(digit, max = 255))


#Tensor Slicing
mySlice <- mnist$train$x[20:90,,]
halfImageSlice <- mnist$train$x[5, 15:28,15:28]

#Creating data batches of size of 128
firstBatch <- mnist$train$x[1:128,,]
secondBatch <- mnist$train$x[129:256,,]

#Matrix addition of different dimension of matrices
sweep(x, 2, y, `+`) #2 is the number of dimensions of x over to which sweep on y 


#Sweep of a 2D and a 4D matrix
# x is a tensor of random values with shape (64, 3, 32, 10)
x <- array(round(runif(1000, 0, 9)), dim = c(64, 3, 32, 10))
# y is a tensor of 5s of shape (32, 10)
y <- array(5, dim = c(32, 10))
# The output z has shape (64, 3, 32, 10) like x. Add tensor 3 and 4 of x to y 
z <- sweep(x, c(3, 4), y, pmax)


#R dot operator between two tensors iif dim(x) == dim(y)
z <- x %*% y

#Tensor Reshaping (array_reshape when from R column to keras row based array)
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28))


#Example
x <- matrix(c(0, 1,
              2, 3,
              4, 5),
            nrow = 3, ncol = 2, byrow = TRUE)
#x <- matrix(c(0,1,2,3,4,5), nrow = 3, ncol = 2)
x <- array_reshape(x, dim = c(6,1))
x <- array_reshape(x, dim = c(2,3))

#Transpose a matrix
x <- matrix(c(round(runif(60000, 0, 100))), nrow = 200, ncol = 300)
x <- t(x)

#Geometric interpretation of tensor operations (imagine being ploted on a (x,y) coord)
A = [0.5, 1.0] 
A = [1, .25]


#The engine of the NN, A relu on the addition of the dot product of W*input plus b(vector)
output = relu(dot(W, input) + b) #Weight, bias

# Naive Reulu function creation
naive_relu <- function(x){
  for (row in dim(x)[1]) 
    for (col in dim(x)[0])
      x[row, col] <- max(x[row,col],0)
}

#Schtocastic Gradient Decent Pseudocode
past_velocity <- 0
momentum <- 0.1
while (loss > 0.01) {
  params <- get_current_parameters()
  w <- params$w
  loss <- params$loss
  gradient <- params$gradient
  
  velocity <- past_velocity * momentum + learning_rate * gradient
  w <- w + momentum * velocity - learning_rate * gradient
  past_velocity <- velocity
  update_parameter(w)
}






