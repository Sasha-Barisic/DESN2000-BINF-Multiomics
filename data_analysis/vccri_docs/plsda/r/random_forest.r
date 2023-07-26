rm(list=ls())

# setwd('C:/Users/user1/Desktop/DESN2000/DESN2000-BINF-Multiomics/data_analysis/vccri_docs/plsda/r')

library(VIM)
library(caret)
library(tidyr)
library(rpart)
library(caTools)
library(randomForest)
library(ropls) 
library(ggforce) 
library(ggplot2) 
library(ggprism) 
library(cowplot)
library(pheatmap)


otu_raw <- read.table(file="in.csv",sep=",",header=T,check.names=FALSE ,row.names=1)
totu_raw <- as.data.frame( t(otu_raw))




totu_raw$class <- rownames(totu_raw)



for(i in 1:6) {
  
  if (grepl("^CZ.", totu_raw[i, 20])) {
    totu_raw[i, 20] = 0
    
  } else {
    totu_raw[i, 20] = 1
  }
}

totu_raw$class <- as.numeric(as.vector(totu_raw$class))
colnames(totu_raw) <- c( paste("x", 1:19, sep = ""), 'class')

set.seed(2023) 

totu_raw$class = factor(totu_raw$class)

index <- createDataPartition(totu_raw$class, p = 0.7)

train <- totu_raw[index$Resample1, ]

test <- totu_raw[-index$Resample1, ]




fitForest <- randomForest(class~., data=train, na.action = na.roughfix, importance = T)

fitForest


jpeg('random_forest_classification.jpeg', width=500, height = 500)

trainerErr <- as.data.frame(plot(fitForest, type = 'l'))
trainerErr

colnames(trainerErr) <- paste('error', colnames(trainerErr), sep='')
trainerErr

trainerErr$ntree <- 1:nrow(trainerErr)
trainerErr

trainerErr <- gather(trainerErr, key='Type',  value='Error', 1:3)
trainerErr

ggplot(trainerErr, aes(x=ntree, y=Error)) +
  geom_line(aes(linetype=Type, color=Type)) +
  ggtitle('random forest classification model') +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()


imp <- importance(fitForest)
imp
rownames(imp) <- substr( rownames(otu_raw), 1, 14 )
colnames(imp) <- c('CZ', 'CL1', 'MeanDecreaseAccuracy', 'MeanDecreaseGini')
imp <- cbind(imp, substr( rownames(imp), 1, 14 ))
colnames(imp) <- c('CZ', 'CL1', 'MeanDecreaseAccuracy', 'MeanDecreaseGini','G')
imp <- as.data.frame(imp)

imp$MeanDecreaseAccuracy <- as.numeric(imp$MeanDecreaseAccuracy) / 200

imp <- imp[order(imp$MeanDecreaseAccuracy, imp$MeanDecreaseGini), ]
imp


p2 <- ggplot(imp,aes(x = MeanDecreaseAccuracy, y = factor(G, levels = G, ordered = TRUE), color=imp$G),size=0.8) +
  geom_point(size=3)+
  labs(x = "MeanDecreaseAccuracy", y = "G", title = NULL)+
  theme_prism(palette = "candy_bright",
              base_fontface = "bold", 
              base_family = "serif", 
              base_line_size = 0.8, 
              axis_text_angle = 0) +
  theme(legend.position = "none") 

p2




hp <- imp
hp$MeanDecreaseAccuracy <- NULL
hp$MeanDecreaseGini <- NULL
hp$G <- NULL
rownames(hp) <- as.matrix(1:19)
hp <- hp[order(hp$CZ, decreasing = T), ]
hp$CZ <- as.numeric(hp$CZ)
hp$CL1 <- as.numeric(hp$CL1)

p3 <- pheatmap(hp,
               cluster_cols = F, cluster_rows = F, scale = "none",
               treeheight_col = 0, treeheight_row = 0,
               display_numbers = F,
               border_color = "black",
               show_rownames =F,
               cellwidth = 12, cellheight = 12
)

p3

jpeg('MeanDecreaseAccuracy.jpeg', width=500, height = 500)
cowplot::plot_grid(p2, p3$gtable, ncol = 2)
dev.off()

