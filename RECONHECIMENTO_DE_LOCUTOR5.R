

install.packages("mxnet")
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

############################## PARAMETROS FUNCAO TRAIN ############################## 

# Define funcao trainControl para validacao cruzada

########## RNA ##########
#Resampling: Cross-Validated (10 fold, repeated 10 times)
ctrl <- trainControl(method = "repeatedcv", number = 10,  verboseIter = TRUE,savePredictions = TRUE)

#mlp_grid1 = expand.grid(layer1 = c(5),.layer2 =c(10),layer3=c(20),.decay = 0.1)
#mlp_grid1 = expand.grid(layer1 = c(2),   
#                        .layer2 = c(4),
#                        .layer3 = 8,
#                        learningrate = c(0.001, 0.0001),
#                        dropout = c(0, 0.2),
#                        beta1 = .9,
#                        beta2 = 0.999,
#                        activation = 'relu')

#mlp_grid1 = expand.grid(.size = c(2), .decay = 0.1)
mlp_grid1 = data.frame(layer1 =c(1:10), layer2 = c(1,10), layer3 =c(1,15))

modelLookup(model = 'neuralnet')
modelLookup(model = 'mlpML')
library(neuralnet)
set.seed(seed)
# start timer
#start_time <- Sys.time()
model_rna1 <- train(label~., 
                    data=dados_treinamento, 
                    method="mlpML",
                    #size = c(5),
                    tuneGrid =mlp_grid1,
                    #Std_Backpropagation, BackpropBatch, BackpropChunk, BackpropMomentum, BackpropWeightDecay, Rprop, Quickprop, SCG (scaled conjugate gradient),
                    #learnFunc = "Std_Backpropagation",
                    #initFunc = "Randomize_Weights", 
                    #initFuncParams = c(-0.3, 0.3),
                    #learnFuncParams = c(0.2, 0),
                    #updateFunc = "Topological_Order", 
                    #updateFuncParams = c(0),
                    #hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = TRUE,
                    #linOut = FALSE, 
                    #inputsTest = NULL, 
                    #targetsTest = NULL,
                    #pruneFunc = NULL, 
                    #pruneFuncParams = NULL,
                    metric=metric,
                    maxit = 30,
                    preProcess = c("center","scale"),
                    trControl=ctrl,
                    learningrate = 0.01,
                    #shufflePatterns = TRUE,
                    threshold = 0.01,     
                    stepmax = 500) 


weightMatrix(model)



#DEEP MXNET
#https://www.r-bloggers.com/deep-learning-with-mxnetr/
#model <- mx.mlp(dados_treinamento_sem_label_scaled, labels_dados_treinamento, hidden_node=10, out_node=2, out_activation="softmax",
#                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
#                eval.metric=mx.metric.accuracy)
#preds = predict(model, dados_teste_sem_label_scaled)
#pred.label = max.col(t(preds))-1
#table(pred.label, labels_dados_teste)


model_rna1
model_rna2
#set.seed(seed)
#model_rna2 <- train(label~., data=dados_treinamento, trControl=ctrl,metric=metric, method="mlp", preProcess = c("center","scale"))
#set.seed(seed)
model_rna3 <- train(label~., data=dados_treinamento, method="mlpML", metric=metric, preProcess = c("center","scale"),trControl=ctrl)


results <- resamples(list("RNA 1"=model_rna1, "RNA 2" = model_rna2)) 



# Compara????o
summary(results)

# Boxplots dos resultados
accuracy_training_models=bwplot(results,pch='|')
plot(accuracy_training_models[1])

########################## RESULTADOS ########################## 

######## RNA
model_rna1
model_rna1$results
predicao_rna1 <- predict(model_rna1, dados_teste_sem_label, type = "prob")
t<-table(predictions = predicao_rna1, actual = labels_dados_teste)
t
prop.table(table(predicao_rna1 == labels_dados_teste))

############ Matriz de confus??o ####################
cfm_rna1 <- confusionMatrix(predicao_rna1, labels_dados_teste)






