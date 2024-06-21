

install.packages("neuralnet")
library(neuralnet)
library(RSNNS)
library(caret)
library(datasets)

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
labels_dados_teste <- dados_teste$label

############################## PARAMETROS FUNCAO TRAIN ############################## 

# Define funcao trainControl para validacao cruzada
library(caret)
library(nnet)
########## RNA ##########
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

set.seed(seed)
model_rna1 <- train(label~., data=dados_treinamento, method="nnet", metric=metric, preProcess = c("center","scale"),trControl=ctrl)
#set.seed(seed)
#model_rna2 <- train(label~., data=dados_treinamento, trControl=ctrl,metric=metric, method="mlp", preProcess = c("center","scale"))
#set.seed(seed)
model_rna3 <- train(label~., data=dados_treinamento, trControl=ctrl,metric=metric, method="mlpML", preProcess = c("center","scale"))


results <- resamples(list("RNA 1"=model_rna1, "RNA 2" = model_rna2, "RNA 3"=model_rna3)) 


# Compara????o
summary(results)

# Boxplots dos resultados
accuracy_training_models=bwplot(results,pch='|')
plot(accuracy_training_models[1])

########################## RESULTADOS ########################## 

######## RNA
model_rna1
model_rna1$results
predicao_rna1 <- predict(model_rna1, dados_teste_sem_label)
t<-table(predictions = predicao_rna1, actual = labels_dados_teste)
t
prop.table(table(predicao_rna1 == labels_dados_teste))

############ Matriz de confus??o ####################
cfm_rna1 <- confusionMatrix(predicao_rna1, labels_dados_teste)






