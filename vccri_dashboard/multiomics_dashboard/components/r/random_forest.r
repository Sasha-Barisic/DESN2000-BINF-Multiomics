rm(list=ls())#clear Global Environment
setwd('./multiomics_dashboard/components/r/')
library(VIM)
library(caret)
library(tidyr)
library(rpart)
library(caTools)
library(randomForest)


otu_raw <- read.table(file="in.csv",sep=",",header=T,check.names=FALSE ,row.names=1)
otu_raw$class <- rownames(otu_raw)
j = length(otu_raw[1,])
for(i in 1:length(otu_raw[,1])) {
  if (grepl("^neg", otu_raw[i, j])) {
    otu_raw[i, j] = 0
  } else {
    otu_raw[i, j] = 1
  }
}

otu_raw$class <- as.numeric(as.vector(otu_raw$class))
set.seed(2023) 
# Splitting data in train and test data
otu_raw$class = factor(otu_raw$class)
index <- createDataPartition(otu_raw$class, p = 0.7)
train <- otu_raw[index$Resample1, ]
test <- otu_raw[-index$Resample1, ]


fitForest <- randomForest(class~., data=train, na.action = na.roughfix, importance = T)
fitForest

jpeg('./results/rf/class.jpeg', width=500, height = 500)

trainerErr <- as.data.frame(plot(fitForest, type = 'l'))
colnames(trainerErr) <- paste('error', colnames(trainerErr), sep='')
trainerErr$ntree <- 1:nrow(trainerErr)
trainerErr <- gather(trainerErr, key='Type',  value='Error', 1:3)
ggplot(trainerErr, aes(x=ntree, y=Error)) +
  geom_line(aes(linetype=Type, color=Type)) +
  ggtitle('random forest classification model') +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()
write.csv(trainerErr, "./results/rf/class.csv", row.names=FALSE)

importance(fitForest)
jpeg('./results/rf/imp.jpeg', width=500, height = 500)
varImpRes <- varImpPlot(fitForest, pch = 20, main='importance of variables')
write.csv(varImpRes, "./results/rf/imp.csv", row.names=TRUE)

dev.off()
