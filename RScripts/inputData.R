#读入数据
raw_dataset <- read.csv("digitRecg_train.csv",header = T,stringsAsFactors = F)
eg <- raw_dataset[10,]
cat(sprintf("数据集中包含的样本个数：%d ",NROW(raw_dataset)))

#将label设置成factor
raw_dataset$label <- as.factor(raw_dataset$label)

#绘制各个数字的样本图形
plotData <- matrix(ncol = NCOL(raw_dataset),nrow = 0)
for(i in 0:9)
{
  plotData <- rbind(plotData,subset(raw_dataset,label == i)[1:10,])
}
plotData <- plotData[,-1]

par(mfcol = c(10,10),mar = rep(0,4))
apply(plotData, 1, plotDigits.eg)
dev.off()

#绘制均值和方差图
par(mfrow = c(2,5),mar = rep(0.2,4))
for(i in 0:9)
{
  x <- apply(subset(raw_dataset,label == i )[,-1],2,mean)
  plotDigits.eg(x)
}
dev.off()

par(mfrow = c(2,5),mar = rep(0.2,4))
for(i in 0:9)
{
  x <- apply(subset(raw_dataset,label == i)[,-1],2,sd)
  plotDigits.eg(x)
}
dev.off()

#提取flag
flag <- raw_dataset[,1]

#显示样本中各个数字占的比例
prop.table(table(flag))

#控制train和test的取值至[0,1]之间，作为dataset1
dataset1 <- raw_dataset[,-1]
VAR <- apply(dataset1, 2, var)
summary(VAR)

#将dataset转换成0-1矩阵，作为dataset2
index <- which(dataset1 > 0,arr.ind = T)
dataset2 <- dataset1
dataset2[index] <- 1

#释放内存空间
rm(i,index,VAR,x)
