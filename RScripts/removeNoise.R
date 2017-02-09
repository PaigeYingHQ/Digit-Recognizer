#ֻ��������������Ϣ�����ݣ���Ϊdataset3
#�����������в���
par(mfrow = c(1,4),mar = rep(0.2,4))

plotDigits.eg(as.numeric(eg[-1]))
eg.test <- removeOuter(as.numeric(eg[-1]),0.75)
plotDigits.eg(eg.test)
eg.test <- removeOuter(as.numeric(eg[-1]),0.5)
plotDigits.eg(eg.test)
eg.test <- removeOuter(as.numeric(eg[-1]),0.25)
plotDigits.eg(eg.test)
dev.off()

#����ȥ��֮��������ֵ�����ͼ��
test <- apply(plotData, 1, function(x){removeOuter(x, prob = 0.5)})
test <- t(test)
par(mfcol = c(10,10),mar = rep(0,4))
apply(as.matrix(test), 1, plotDigits.eg)
dev.off()

#Ӧ����ԭʼ���ݼ�ȥ��
dataset3 <- apply(raw_dataset[,-1], 1, function(x){removeOuter(x, prob = 0.5)})
dataset3 <- t(dataset3)
dataset3 <- as.data.frame(cbind(label = as.numeric(as.character(flag)), dataset3))

#ʩ�Ų����ڴ�
rm(raw_dataset)

#����ȥ���ľ�ֵ�ͷ���ͼ
par(mfrow = c(2,5),mar = rep(0.2,4))
for(i in 0:9)
{
  x <- apply(subset(dataset3,label == i)[,-1],2,mean)
  plotDigits.eg(x)
}
dev.off()

par(mfrow = c(2,5),mar = rep(0.2,4))
for(i in 0:9)
{
  x <- apply(subset(dataset3,label == i)[,-1],2,sd)
  plotDigits.eg(x)
}
dev.off()

#��dataset3ת����0-1������Ϊdataset4
index <- which(dataset3 > 0,arr.ind = T)
dataset4 <- dataset3
dataset4[index] <- 1

rm(eg.test,test,i,x,index)
rm(plotData)

#������Ӧ�����ݼ���ע���޸���Ӧ�ı���Ŀ¼��
save.image("D:/R/DR/allDataset.RData")

rm(dataset2,dataset3,dataset4)

#����ֵ�ֽ�
library(sp)
library(raster)
library(jpeg)

#�ҵ��ֽ�������
s <- proc.time()
model.set1.svd <- svd(dataset1)
print(s-proc.time())

#print(s-proc.time())
#�û�    ϵͳ    ���� 
#-149.45   -0.81 -153.69

k <- seq(from = 50, to = 250, by = 50)
Kvect <- svd.setK(model.set1.svd,k,dataset1)

plot(Kvect)
dataset1.svd <- recovery(model.set1.svd,k = 250,dataset1)

save.image("D:/R/DR/set1_svd.RData")