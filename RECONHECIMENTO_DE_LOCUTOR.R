install.packages("neuralnet")
library(neuralnet)
# adapted iris
setwd("C:/Program Files/RStudio")
library(datasets)
data(iris)
data <-read.csv("gender_voice_dataset.csv", header = T, sep = ",", dec = ".")
data2 <- data

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








