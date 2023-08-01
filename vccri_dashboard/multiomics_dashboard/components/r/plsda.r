
rm(list=ls())#clear Global Environment
setwd('./multiomics_dashboard/components/r')
library(ropls) 
library(ggforce) 
library(ggplot2) 
library(ggprism) 
library(cowplot)

otu_raw <- read.table(file="in.csv",sep=",",header=T,check.names=FALSE ,row.names=1)
group <- read.table(file="in_g.csv",sep=",",header=T,check.names=FALSE ,row.names=1)

otu <- t(otu_raw)
df1_oplsda <- opls(otu, group$label, predI = 1, orthoI = NA,crossvalI=6)

data <- as.data.frame(df1_oplsda@scoreMN)
o1 <- df1_oplsda@orthoScoreMN[,1]
data$o1 <- o1
data$group = group$label
data$samples = rownames(data)
x_lab <- df1_oplsda@modelDF[1, "R2X"] * 100

write.csv(data, "./results/ortho/score.csv", row.names=FALSE)

col=c("#1597A5","#FFC24B")
p1 <- ggplot(data,aes(x=p1,y=o1,color=group))+
  theme_bw()+
  geom_point(size=6)+
  theme(panel.grid = element_blank())+
  geom_vline(xintercept = 0,lty="dashed",color="red")+
  geom_hline(yintercept = 0,lty="dashed",color="red")+
  
  labs(x=paste0("P1 (",x_lab,"%)"),
       y=paste0("to1"))+
  stat_ellipse(data=data,
               geom = "polygon",level = 0.95,
               linetype = 2,size=0.5,
               aes(fill=group),
               alpha=0.2,
               show.legend = T)+
  scale_color_manual(values = col) +
  scale_fill_manual(values = c("#1597A5","#FFC24B"))+
  theme(axis.title.x=element_text(size=12),
        axis.title.y=element_text(size=12,angle=90),
        axis.text.y=element_text(size=10),
        axis.text.x=element_text(size=10),
        panel.grid=element_blank())

jpeg('./results/ortho/score.jpeg', width=500, height = 500)
p1
dev.off()

data_VIP <- df1_oplsda@vipVn
data_VIP_select <- data_VIP

data_VIP_select <- cbind(otu_raw[names(data_VIP_select), ], data_VIP_select)
data_VIP_select
names(data_VIP_select)[7] <- "VIP"

data_VIP_select$G = rownames(data_VIP_select) 

data_VIP_select <- data_VIP_select[order(data_VIP_select$VIP, data_VIP_select$G), ]

p2 <- ggplot(data_VIP_select,aes(x = VIP, y = factor(G, levels = G, ordered = TRUE), color=data_VIP_select$G),size=0.8) +
  geom_point(size=3)+
  labs(x = "VIP", y = "G", title = NULL)+
  theme_prism(palette = "candy_bright",
              base_fontface = "bold", 
              base_family = "serif", 
              base_line_size = 0.8, 
              axis_text_angle = 45) +
  theme(legend.position = "none")

jpeg('./results/ortho/vip.jpeg', width=1000, height = 1000)
p2
dev.off()

write.csv(data_VIP_select, "./results/ortho/vip.csv", row.names=FALSE)
