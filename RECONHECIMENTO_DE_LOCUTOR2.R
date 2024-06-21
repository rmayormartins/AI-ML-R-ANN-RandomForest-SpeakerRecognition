install.packages("neuralnet")
library(neuralnet)
# adapted iris
setwd("C:/Program Files/RStudio")
library(datasets)
#data(iris)
data <-read.csv("gender_voice_dataset.csv", header = T, sep = ",", dec = ".")

#shuffle the vector
data2 <- data[sample(1:nrow(data),length(1:nrow(data))),1:ncol(data)]
dataValues <- data[,1:20]
library(RSNNS)
dataTargets <- decodeClassLabels(data[,21])
#irisTargets <- decodeClassLabels(iris[,5], valTrue=0.9, valFalse=0.1)

data <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.15)
data <- normTrainingAndTestSet(data)

model <- mlp(data$inputsTrain, data$targetsTrain, size=c(5,3,2), learnFuncParams=c(0.1), 
             maxit=50, inputsTest=data$inputsTest, targetsTest=data$targetsTest)

summary(model)
model
weightMatrix(model)
extractNetInfo(model)
neuralweights(model)

library(devtools)
library(reshape2)
library(nnet)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(model)

# pruned model using code from RSSNS pruning demo
pruneFuncParams <- list(max_pr_error_increase = 10.0, pr_accepted_error = 1.0, 
                        no_of_pr_retrain_cycles = 1000, min_error_to_stop = 0.01, init_matrix_value = 1e-6, 
                        input_pruning = TRUE, hidden_pruning = TRUE)

model <- mlp(data$inputsTrain, data$targetsTrain, size=c(5,3,2), learnFuncParams=c(0.1), 
             maxit=50, inputsTest=data$inputsTest, targetsTest=data$targetsTest,pruneFunc = "OptimalBrainSurgeon", 
             pruneFuncParams = pruneFuncParams)

library(devtools)
library(reshape2)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#erro dando aqui
plot.nnet(model)


#parei aqui
par(mfrow=c(2,2))
plotIterativeError(model)

predictions <- predict(model,data$inputsTest)
plotRegressionError(predictions[,2], data$targetsTest[,2])
confusionMatrix(data$targetsTrain,fitted.values(model))
confusionMatrix(data$targetsTest,predictions)

plotROC(fitted.values(model)[,2], data$targetsTrain[,2])
plotROC(predictions[,2], data$targetsTest[,2])
#' 
#confusion matrix with 402040-method
confusionMatrix(data$targetsTrain, encodeClassLabels(fitted.values(model),method="402040", l=0.4, h=0.6))





############################################################




#-------------------ANTIGO------------------------
#ESSE PEDAÇO ERA SO PRA PEGAR O HEADER, POR ISSO ESTA F
#data3 <-read.csv("gender_voice_dataset.csv", header = F, sep = ",", dec = ".")
#names(data3) <- data3[1,]


data2$male <- c(data2$label == 'male')
data2$female <- c(data2$label == 'female')
# iris2$Species <- NULL

# training and validation subsets
train.samples <- sample(nrow(data), nrow(data)*0.5)
train <- data2[train.samples,]
valid <- data2[-train.samples,]

#ARRUMAR OS NOMES DAS COLUNAAAAAASS

# fit model
inet <- neuralnet(male + female ~ meanfreq + sd + median + Q25 + Q75 + IQR + skew + kurt + sp.ent + sfm + mode + centroid + meanfun + minfun + maxfun + meandom + mindom + maxdom + dfrange + modindx,
train, hidden=c(5,3,2),rep=2, lifesign="full",algorithm = "rprop+")

inet$net.result
inet$result.matrix
head(inet$generalized.weights[[1]])

#ARRUMAR ISSOOOOO
# prediction 
pred <- compute(inet, valid[,1:20])
head(pred$net.result) # only one level (probability of each category)
predspp <- factor(c("male" , "female"))[apply(pred$net.result, MARGIN=1, FUN=which.max)]
table(predspp, valid$label)

tab <- table(predspp, valid$label)
library(caret)

confusionMatrix(tab)








