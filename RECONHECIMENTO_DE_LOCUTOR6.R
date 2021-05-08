

#install.packages("mxnet")
library(neuralnet)
library(RSNNS)
library(caret)
library(datasets)
library(nnet)
library(mxnet)

############################## DATASET ############################# 
seed <- 123
metric <- "Accuracy"

setwd("C:/Program Files/RStudio")
tabela <-read.csv("gender_voice_dataset.csv", header = T, sep = ",", dec = ".")
tabela <-na.omit(tabela)
tabelaClass <- tabela[21]

# Apresenta quantidade respectiva as classes de veiculos presentes no conjunto de dados
#summary(tabela)
#table(tabela$Class)

## Preparacao dos dados: Separa 80% para treinamento e 20% para teste
set.seed(seed)
grupos <- sample(2, nrow(tabela), replace=TRUE, prob=c(0.8, 0.2))
dados_treinamento <- tabela[grupos==1,]
dados_teste <- tabela[grupos==2,]

#library(caret)   caso usar o normalizeData
# Remove a coluna que identifica a classe de cada veiculo em ambos os grupos
dados_treinamento_sem_label <- dados_treinamento[-21]
#dados_treinamento_sem_label_normalizado <-  normalizeData(dados_treinamento_sem_label, "norm")
dados_treinamento_sem_label_scaled = as.data.frame(lapply(dados_treinamento_sem_label, scale))
dados_teste_sem_label <- dados_teste[-21]
dados_teste_sem_label_scaled = as.data.frame(lapply(dados_teste_sem_label, scale))

# Guarda a classe que identifica cada veiculo em ambos os grupos
labels_dados_treinamento <- dados_treinamento$label
labels_dados_treinamento_decode <- decodeClassLabels(labels_dados_treinamento)
labels_dados_teste <- dados_teste$label
labels_dados_teste_decode <- decodeClassLabels(labels_dados_teste)

#entrar na lista usar [[2]] por exemplo

############################## PARAMETROS FUNCAO TRAIN ############################## 


########## RNA 1 ##########

# create empty data frame 
learnCurve <- data.frame(m = integer(21),
                         trainACC = integer(21),
                         )

# Define funcao trainControl para validacao cruzada
#Resampling: Cross-Validated (10 fold, repeated 10 times)
# Run algorithms using 10-fold cross validation with 3 repeats
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10,
                     classProbs= TRUE,
                     verboseIter = TRUE,
                     savePredictions = TRUE,
                     returnResamp ="all", 
                     returnData=TRUE,
                     summaryFunction = defaultSummary,
                     selectionFunction = "best")

#organiza as 3 layers
#mlp_grid1 = data.frame(layer1 =(1:20), layer2 =(1:20), layer3 =(1:20))
mlp_grid1 = data.frame(layer1 =1, layer2 =1, layer3 =1)


#modelLookup(model = 'neuralnet')
modelLookup(model = 'mlpML')

set.seed(seed)
# start timer
#start_time <- Sys.time()

# loop over training examples
#for (i in 1:5) {
#  learnCurve$m[i] <- i


model_rna1 <- train(label~., 
                    data=dados_treinamento, 
                    method="mlpML",
                    tuneGrid =mlp_grid1,
                    linout=TRUE,
                    trace=TRUE,
                    verboseIter=TRUE,
                    #Std_Backpropagation, BackpropBatch, BackpropChunk, BackpropMomentum, BackpropWeightDecay, Rprop, Quickprop, SCG (scaled conjugate gradient),
                    learnFunc = "Std_Backpropagation",
                    initFunc = "Randomize_Weights", 
                    #initFuncParams = c(-0.3, 0.3),
                    #learnFuncParams = c(0.2, 0),
                    #updateFunc = "Topological_Order", 
                    #updateFuncParams = c(0),
                    hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = TRUE,
                    #linOut = FALSE, 
                    #inputsTest = NULL, 
                    #targetsTest = NULL,
                    #pruneFunc = NULL, 
                    #pruneFuncParams = NULL,
                    metric=metric,
                    maxit = 100,
                    tuneLength = "",
                    preProcess = c("center","scale"), #com pca
                    trControl=ctrl,
                    learningrate = 0.01,
                    threshold = 0.01,
                    stepmax = 1e+05)

#learnCurve$trainACC[i] <- model_rna1$results$Accuracy
#}

#In the neural network terminology:
#one epoch = one forward pass and one backward pass of all the training examples
#batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
#number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
#Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

getModelInfo(model_rna1)

model_rna1$results
model_rna1$resample

model_rna1$results["Accuracy"]

plot(log(learnCurve$trainACC),type = "o",col = "red", xlab = "EPOCH",
     ylab = "ACC", main = "Linear Model Learning Curve")

install.packages("NeuralNetTools")
library("NeuralNetTools")

plotnet(model_rna1)

print.train(model_rna1)
model_rna1$results
model_rna1$resample
df_acc <- model_rna1$resample
df_acc$Accuracy

plot(df_acc$Accuracy,type="b",frame = FALSE, pch=19)

model_rna1

histogram(model_rna1)
densityplot(model_rna1)
resampleHist(model_rna1)
summary(model_rna1)



class(model_rna1)
attributes(model_rna1)
model_rna1$finalModel
class(model_rna1$finalModel)

getModelInfo(model_rna1)
model_rna1$results
model_rna1
summary(model_rna1)
print(model_rna1)

print(model_rna1, printCall = TRUE, details =TRUE, showSD = TRUE)
print(model_rna1$finalModel$archParams)

model_rna1$finalModel[1]
model_rna1$metric

########## RNA 2 ##########
control <- trainControl(method = "repeatedcv", number = 10, repeats = 1, savePredictions = T, classProbs = T, returnResamp = "all")

set.seed(seed)
model_rna2 <- train(label~., data=dados_treinamento,trControl=control,metric=metric, method="nnet", preProcess = c("center","scale"))
plot(model_rna2)
ggplot(model_rna2)
model_rna2$results
model_rna2$resample
model_rna2


########## RNA 3 ##########
ctrl <- trainControl(method = "repeatedcv", number = 10,classProbs= TRUE,  verboseIter = TRUE,savePredictions = TRUE,returnResamp ="all", repeats=3)

pruneFuncParams <- list(max_pr_error_increase = 10.0, pr_accepted_error = 1.0, 
                        no_of_pr_retrain_cycles = 1000, min_error_to_stop = 0.01, init_matrix_value = 1e-6, 
                        input_pruning = TRUE, hidden_pruning = TRUE) 
mlp_grid3 = data.frame(layer1 =1, layer2 =0, layer3 =0)


set.seed(seed)
model_rna3 <- train(label~., 
                    data=dados_treinamento, 
                    method="mlpML",
                    tuneGrid =mlp_grid3,
                    linout=TRUE,
                    trace=TRUE,
                    verbose=TRUE,
                    #Std_Backpropagation, BackpropBatch, BackpropChunk, BackpropMomentum, BackpropWeightDecay, Rprop, Quickprop, SCG (scaled conjugate gradient),
                    learnFunc = "Std_Backpropagation",
                    initFunc = "Randomize_Weights", 
                    #initFuncParams = c(-0.3, 0.3),
                    #learnFuncParams = c(0.2, 0),
                    #updateFunc = "Topological_Order", 
                    #updateFuncParams = c(0),
                    hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = TRUE,
                    #linOut = FALSE, 
                    #inputsTest = NULL, 
                    #targetsTest = NULL,
                    pruneFunc = "OptimalBrainSurgeon", 
                    pruneFuncParams = pruneFuncParams,
                    metric=metric,
                    maxit = 30,
                    tuneLength = 8,
                    preProcess = c("center","scale"), #com pca
                    trControl=ctrl,
                    learningrate = 0.01,
                    threshold = 0.01,     
                    stepmax = 500) 

model_rna3
plot(model_rna3)
ggplot(model_rna3)
library(RSNNS)
extractnetinfo(model_rna3)

##########Comparação RNAS############


#model_rna2
#set.seed(seed)
#model_rna2 <- train(label~., data=dados_treinamento, trControl=ctrl,metric=metric, method="mlp", preProcess = c("center","scale"))
#set.seed(seed)
#model_rna3 <- train(label~., data=dados_treinamento, method="mlpML", metric=metric, preProcess = c("center","scale"),trControl=ctrl)


results <- resamples(list("RNA 1"=model_rna1, "RNA 2" = model_rna2)) 
results2 <- list("RNA 1"=model_rna1, "RNA2"= model_rna2) #so pra teste

dotplot(results)
densityplot(results,
            metric = "Accuracy",
            auto.key = TRUE,
            pch = "|")
bwplot(results,
       metric = "Accuracy")

xyplot(results)

resampleHist(model_rna1)



# Comparacao
summary(results)

# Boxplots dos resultados
accuracy_training_models=bwplot(results,pch='|')
plot(accuracy_training_models[1])

########################## RESULTADOS ########################## 

######## RNA 1
model_rna1
model_rna1$results
predicao_rna1 <- predict(model_rna1, dados_teste_sem_label)
predicao_lista <- predict(results2) #pra ver o que da
t1<-table(predictions = predicao_rna1, actual = labels_dados_teste)
t1
prop.table(table(predicao_rna1 == labels_dados_teste))

######### RNA 2
predicao_rna2 <- predict(model_rna2, dados_teste_sem_label)
predicao_lista <- predict(results2) #pra ver o que da
t2<-table(predictions = predicao_rna2, actual = labels_dados_teste)
t2
prop.table(table(predicao_rna2 == labels_dados_teste))

######### RNA 3
predicao_rna3 <- predict(model_rna3, dados_teste_sem_label)
predicao_lista <- predict(results2) #pra ver o que da
t3<-table(predictions = predicao_rna2, actual = labels_dados_teste)
t3
prop.table(table(predicao_rna3 == labels_dados_teste))

####################### Matriz de confusao ####################
cfm_rna1 <- confusionMatrix(predicao_rna1, labels_dados_teste,mode = "everything")
cfm_rna1
cfm_rna2 <- confusionMatrix(predicao_rna2, labels_dados_teste,mode = "everything")
cfm_rna2
cfm_rna3 <- confusionMatrix(predicao_rna3, labels_dados_teste,mode = "everything")
cfm_rna3


results_final <- list("RNA 1"=cfm_rna1$overall[1], "RNA2"= cfm_rna2$overall[1], "RNA3" = cfm_rna3$overall[1])

library (plyr)
df <- ldply (results_final, data.frame)
df <- rename(df, c(".id"="Technique","X..i.."="Acc"))
df
#df$Technique <- rownames(df)
df$Technique

plot(df$Acc,type="b",axes=FALSE,pch=19)
axis(1,at=seq(df$Technique),labels=df$Technique) #to draw x axis 
axis(2)  #to draw y axis .since your y axis is well computed dont need to use 'at'
grid()



extractPrediction(results2, dados_teste_sem_label)
extractProb(results2, dados_teste_sem_label)

#testar essa comparação
model_comp <- as.data.frame(
  rbind(
    summod(cm_knn1, knn_model1),
    summod(cm_knn2, knn_model2),
    summod(cm_knn3, knn_model3)))
rownames(model_comp) <- c("Model 1", "Model 2", "Model 3")
pander(model_comp[,-3], split.tables=Inf, keep.trailing.zeros=TRUE,
       style="rmarkdown",
       caption="Model results when comparing predictions and test set")
