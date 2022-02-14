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
library(mltools)
library(keras)
library(sparklyr)
#library(imager) Won't work
library(png) 
library(RCurl)

#####################Chapter 2 - Getting Started With NN

#use_condaenv("r-tensorflow")
#tensorflow::install_tensorflow()

#Using the boston housing data set (training and x)
boston <- keras::dataset_boston_housing()
#Using the training data
data <- boston$train$x


#Hold out validation data
#80 percent of the data in indices
indices <- sample(1:nrow(data), size = 0.80 * nrow(data))

#Evaluation and Training data
evaluation_data <- data[-indices,]
training_data <- data[indices,]

get_model <- function(){
  model <- keras::keras_model_sequential() %>%
    keras::layer_dense(units = 64, activation = "relu", input_shape = nrow(data)) %>%
    keras::layer_dense(units = 64, activation = "relu") %>%
    keras::layer_dense(units = 1)
  
  model %>% 
    keras::compile(
      optimizer = "cross_entropy",
      loss = "mse",
      metrics = c("accuracy")
    )
}
#Get the model
model <- get_model()
#Taining on the training data
model %>% train(training_data)
#evaluating on the validation data
validation_score <- model %>% evaluate(validation_data)
model <- get_model()
model %>% train(data)
#Evaluating on the test data
test_score <- model %>% evaluate(test_data)


#K-Fold Cross-Validation
k <- 4
#Indices
indices <- sample(1:nrow(data))
#Getting indices into groups to number of k folds
folds <- cut(indices, breaks = k, labels = FALSE)
#To store k fold results
validation_scores <- c()

for (i in 1:k) {
  #indices for whom to train
  validation_indices <- which(folds == i, arr.ind = TRUE)
  #Val and training data
  validation_data <- data[validation_indices,]
  training_data <- data[-validation_indices,]
  
  model <- get_model()
  model %>% train(training_data)
  results <- model %>% evaluate(validation_data)
  validation_scores <- c(validation_scores, results$accuracy)
}
validation_score <- mean(validation_scores)
model <- get_model()
model %>% train(data)
results <- model %>% evaluate(test_data)


#Feature-wise normalization in 2D arrays no need to feed mean and std for each
# attribute
x <- scale(data)

#Because we want to normalize the train and test we find mean and std from training data
mean <- apply(data, 2, mean)
std <- apply(data,2,sd)

#Then nomalized
normalized_train_data <- scale(data, scale = std, center = mean)
normalized_test_data <- scale(boston$test$x, center = mean, scale = std)



#Original model 
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
#Model version with lower capacity
model <- keras_model_sequential() %>%
  layer_dense(units = 4, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Sometimes is good to have smaller number of layers


#Version of the model with a much higher capacity
model <- keras_model_sequential() %>%
  layer_dense(units = 514, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 514, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#L2 Weight regulization to the model (.001 means )
#every coefficient in the weight matrix of the layer will add 0.001 * weight_coefficient_value 
#to the total loss of the network
model <- keras_model_sequential() %>%
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001), activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Different weight regulazers (less septible to overfitting)
regularizer_l1(0.001)
regularizer_l1_l2(l1 = 0.001, l2 = 0.001)









