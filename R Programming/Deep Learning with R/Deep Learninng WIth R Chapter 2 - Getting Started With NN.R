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

#Previous path
# "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/Library/Frameworks/Mono.framework/Versions/Current/Commands:/Applications/RStudio.app/Contents/MacOS/postback"

#use_condaenv("r-tensorflow")
#tensorflow::install_tensorflow()

#Fully connected layer (output = 32 & input = c(12,12)). For simple vector data
layer_dense(units = 32, input_shape = c(12,12))
#Recurrent layers. For 3D tensors ("samples", "timesptep", "feature")
layer_lstm()
#2D Convolution layer. For 4D tensors
layer_conv_2d()


#Layer that only accepts a 2D  with input dim of 784 and returns a dim of 32
layer <- layer_dense(units = 32, input_shape = c(784))


#For the second layer no need to set input to 32 as it knows
model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = c(784)) %>%
  layer_dense(units = 32)


#The most common method of architeture
model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = "softmax")

#Another method of creating the model
input_tensor <- layer_input(shape = c(784))
output_tensor <- input_tensor %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model <- keras_model(inputs = input_tensor, outputs = output_tensor)

#Compiling the model, setting the loss, optimized and the metrics
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.0001),
  loss = "mse",
  metrics = c("accuracy")
)

#To actually train the model, with batch of 128 and 10 cycles
model %>% 
  fit(input_tensor, target_tensor, batch_size = 128, epochs = 10)





######  The IMBD Data Set (Binary Classification)  ######## 
###########################################################

imbd <- keras::dataset_imdb(num_words = 10000)

train_data <- imbd$train$x
train_labels <- imbd$train$y
test_data <- imbd$test$x
test_labels <- imbd$test$y

#str of the first movie review (integers are coded for specific words)
str(train_data[1])

#Moview review 1 being positive and 0 a negative review
train_labels[1]

#BC we use only 10,000 words no letter would be avobe 10,000
max(sapply(train_data, max))

#To decode one integer to english
word_index <- dataset_imdb_word_index() #Get word index
reverse_word_index <- names(word_index) #Give the string words
names(reverse_word_index) <- word_index #Give the index to be as names

#Example for first review
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

#Encoding the integers sequence into a binary matrix
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
 


#Example for one moview review
str(x_train[1,])

#Converting labels to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


##Building the network
#Dense layer and relu for 1s and 0s as the input vector data
#output = relu(dot(W, input) + b) #Sample NN calculation e.i
layer_dense(units = 16, activation = "relu")

#Model Definition
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% keras::compile(
  loss = "binary_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)

#Setting aside training validation data aka. (more training data partition)
val_indices <- 1:10000
x_val <- x_train[val_indices,]
y_val <- y_train[val_indices]
partial_x_train <- x_train[-val_indices,]
partial_y_train <- y_train[-val_indices]

#Training the model
history <- model %>% keras::fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


#The history Object
str(history)
  
#Plotting the training and validation
plot(history)

###RETRAINING WITH LESS EPOCHS SINCE IT PEAKED AT 4 ON OUR VALIDATION SET
modelOne <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

modelOne %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

modelOne %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
resultsOne <- modelOne %>% evaluate(x_test, y_test)
results <- model %>% keras::evaluate(x_test,y_test)

#Predict On New Data
predictedTest <- modelOne %>%
  predict(x_test[1:10,])

#Rond the probaility ratios
roundedPredictedTest <- round(predictedTest)

#Confusion Matrix
CM <- confusionMatrix(as.factor(roundedPredictedTest), as.factor(y_test[1:10]))
CM$overall

#For all 25,000 test Reviews
allPredictedTest <- modelOne %>%
  predict(x_test)
allRoundedPredictedTest <- round(allPredictedTest)
CM <- confusionMatrix(as.factor(allRoundedPredictedTest), as.factor(y_test))
CM$overall

#We can change to 1 or 3 layers
#We can change from 32 hidden units to 6
#We can change the loss to mse from binary_crossentropy
#We can change the activation fun to tanh from relu

#############    Aditional   #################
######  Configuring the optimizer ei.  ######        
model %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

###Using costume loss and metrics
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = loss_binary_crossentropy,
  metrics = metric_binary_accuracy
)
#################################################






#######  The Reuters Data Set (Multiclassification)  ######
###########################################################

#Get reuters data
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

#Length of how many short newswire are on the training and test data sets
length(train_data)
length(test_data)

#View of the first newswire from the train data (list of indices from dicc)
train_data[[1]] #Ignore 0 , 1 & 2


#Decoding the newswire back to text
word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decoded_newswire <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

#Encoding the data
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

#One hot encoding
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]]] <- 1
  results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)


###### Hot Encoding Self Example ########
############################################################
my_one_hot <- function(data, dimension = 46){
  result <- matrix(0, nrow = length(data), ncol = dimension)
  for (i in 1:length(data)) {
    result[i, data[[i]]] <- 1
  }
  result
}
  
sampleTrain <- my_one_hot(train_labels)
sampleTest <- my_one_hot(test_labels)
############################################################


#Another way as hot encoding usaing keras to categorical 
#(for integer sequence no lists, ei sinlge column frames)
one_hot_train_labels <- keras::to_categorical(train_labels)
one_hot_test_labels <- keras::to_categorical(test_labels)


#Model Definition
model <- keras::keras_model_sequential() %>%
  keras::layer_dense(units = 64, input_shape = c(10000), activation = "relu") %>%
  keras::layer_dense(units = 64, activation = "relu") %>%
  keras::layer_dense(units = 46, activation = "softmax")

#Compile the model
model %>%
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = "rmsprop",
    metrics = c("accuracy")
  )

#Setting Validation data
val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[-val_indices,]

#Training the Model
history <- model %>% keras::fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

#Plotting the history
plot(history)

##########  Retraining a model from scratch
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
results <- model %>% evaluate(x_test, one_hot_test_labels)
results
####################  End of Retraining a model from scratch

#Naive Null approach (guessing)
test_labels_copy <- test_labels
test_labels_copy <- sample(test_labels_copy)
length(which(test_labels == test_labels_copy)) / length(test_labels)


#Generating predictions for new data
predictions <- model %>% predict(x_test)

dim(predictions)
sum(predictions[1,]) #total probalities for a newswire should cum to 1

#To view the corresponding category number
which(predictions[1,] == max(predictions[1,]))
which.max(predictions[1,])

#We get the categorical number for all the newswires since labels range 0-45
sol <- c(sample(x=0, size = dim(predictions)[1], replace = TRUE))
newFinal <- as.data.frame(predictions)
names(newFinal) <- levels(as.factor(test_labels))

for (i in 1:dim(predictions)[1]) {
  sol[i] <- which.max(newFinal[i,]) - 1
}

TP <- sum(test_labels == sol)
FP <- length(sol) - TP
modelAccuracy <- TP/length(sol)
modelAccuracy


#Different way of handling the compile part
model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)




#######  The Boston Housing Price Data Set (Regression)  ######
###############################################################


dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

#Structure of the train data
str(train_data)
str(train_targets)

#Normalize the data 
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
#Applying the scale function
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


#Model definition
build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
    #Single unit and not activation since it is regression and trying to 
    #estimate a single continuous value
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}


#K-Fold Validation for small data sets
k <- 4

#Getting the indices at ramdom
indices <- sample(1:nrow(train_data))
#Assingning the group to a set of indices
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 100
all_scores <- c()

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  #Indices of the values that k fold matches with the corresponding one
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  #Validation data and targets
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  #Partial (e.i rest of the data)
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  #Building the model
  model <- build_model() 
  
  #Training the model (going one by one batch size)
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
  
  #Evaluating the model with the validation data
  results <- model %>%
    evaluate(val_data, val_targets, verbose = 0)
  
  all_scores <- c(all_scores, results[2]) #results[2] -> mean_absolute_error
}

#Scores
all_scores
#Mean of Scores
mean(all_scores)


#Saving the validation logs at each fold
num_epochs <- 500
all_mae_histories <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]

  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  
  mae_history <- history$metrics$val_mae
  
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}


#Building the history of successive mean K-fold validation scores
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

#Listing 3.32 Plotting validation scores
library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

#With geom_smooth now
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

#Which Epoch had the lowest validation mae value
best_epoch = average_mae_history[which.min(average_mae_history$validation_mae),]


#Training the final model
model <- build_model()

model %>% keras::fit(
  train_data,
  train_targets,
  epoch = best_epoch$epoch,
  batch_size = 16,
  verbose = 0
)

results <- model %>% keras::evaluate(
  test_data,
  test_targets
)

#We are still off by $2,715.70 dollars for each house
results


