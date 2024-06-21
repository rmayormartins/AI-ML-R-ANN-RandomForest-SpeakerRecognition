install.packages("neuralnet")
library(neuralnet)
# adapted iris
setwd("C:/Program Files/RStudio")
library(datasets)
#data(iris)
data <-read.csv("gender_voice_dataset.csv", header = T, sep = ",", dec = ".")

library(RSNNS)
library(caret)

#normalizar
data <-  normalizeData(data[,1:20], "norm")
#shuffle
data <-  data[sample(1:nrow(data) ,length(1:nrow(data))), 1:ncol(data)] 


#SPLIT (0.7 train e 0.3 test)
ind <- sample(2, nrow(data), replace=TRUE, prob=c(0.7, 0.3))

trainData <- data[ind==1,]
testData <- data[ind==2,]
#removing factor variable from training and test datasets
#trainData1 <- trainData[-21]
#testData1 <- testData[-21]


#Spet 2: storing target variable for testing and training data.
data_train_labels <- trainData$label 
data_test_labels <- testData$label

#ESSA PARTE PARECE BEM SIMPLES!
samples <- sample(nrow(data), nrow(data)*0.5)
train <- data[samples,]

TrainData <- train[,1:20]
TrainClasses <- train[,21]

#DecTrainClasses <-  decodeClassLabels(TrainClasses)
test <- data[-samples,]
TestData <- test[,1:20]
TestClasses <- test[,21]

set.seed(80)
pruneFuncParams <- list(max_pr_error_increase = 10.0, pr_accepted_error = 1.0, 
                        no_of_pr_retrain_cycles = 1000, min_error_to_stop = 0.01, init_matrix_value = 1e-6, 
                        input_pruning = TRUE, hidden_pruning = TRUE) 
#pruneFunc = "OptimalBrainSurgeon", pruneFuncParams = pruneFuncParams) 

#The model is then built with: 
model <-  mlp(pca.train, TrainClasses, 
              size = c(5,3), 
              maxit = 50,
              initFunc="Randomize_Weights", 
              learnFunc="Std_Backpropagation", 
              learnFuncParams = c(0.1),
              hiddenActFunc = "Act_Logistic",
              shufflePatterns=TRUE,
              linOut=TRUE,
              pruneFunc = "OptimalBrainSurgeon", 
              pruneFuncParams = pruneFuncParams)











#Algorithms in RSNNS for weight pruning include magnitude-based pruning, optimal brain damage, and optimal brain surgeon, whereas algorithms for node pruning include skeletonization and the non-contributing units method

model
summary(model)
model
weightMatrix(model)
extractNetInfo(model)
neuralweights(model)

predictions <-  predict(model, TestData)
par(mfrow=c(2,2))
#Some error plots can be shown:
plotIterativeError(model) 
plotRegressionError(predictions[,2], iris$targetsTest[,2], pch = 3) 
plotROC(fitted.values(model)[,2], iris$targetsTrain[,2]) 
plotROC(predictions[,2], iris$targetsTest[,2])

library(devtools)
library(NeuralNetTools)
plotnet(model)
plotnet(model,prune_col = 'lightblue',prune_lty = "dashed")
weightMatrix(model)
print(model)
table(predspp, valid$Species)

tab <- table(predspp, valid$Species)
library(caret)

confusionMatrix(tab)


confusionMatrix(TestClasses, predictions)






