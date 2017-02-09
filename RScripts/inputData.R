#��������
raw_dataset <- read.csv("digitRecg_train.csv",header = T,stringsAsFactors = F)
eg <- raw_dataset[10,]
cat(sprintf("���ݼ��а���������������%d ",NROW(raw_dataset)))

#��label���ó�factor
raw_dataset$label <- as.factor(raw_dataset$label)

#���Ƹ������ֵ�����ͼ��
plotData <- matrix(ncol = NCOL(raw_dataset),nrow = 0)
for(i in 0:9)
{
  plotData <- rbind(plotData,subset(raw_dataset,label == i)[1:10,])
}
plotData <- plotData[,-1]

par(mfcol = c(10,10),mar = rep(0,4))
apply(plotData, 1, plotDigits.eg)
dev.off()

#���ƾ�ֵ�ͷ���ͼ
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

#��ȡflag
flag <- raw_dataset[,1]

#��ʾ�����и�������ռ�ı���
prop.table(table(flag))

#����train��test��ȡֵ��[0,1]֮�䣬��Ϊdataset1
dataset1 <- raw_dataset[,-1]
VAR <- apply(dataset1, 2, var)
summary(VAR)

#��datasetת����0-1������Ϊdataset2
index <- which(dataset1 > 0,arr.ind = T)
dataset2 <- dataset1
dataset2[index] <- 1

#�ͷ��ڴ�ռ�
rm(i,index,VAR,x)