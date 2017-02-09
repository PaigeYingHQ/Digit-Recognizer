#只保留部分像素信息的数据，作为dataset3
#单个样本进行测试
par(mfrow = c(1,4),mar = rep(0.2,4))

plotDigits.eg(as.numeric(eg[-1]))
eg.test <- removeOuter(as.numeric(eg[-1]),0.75)
plotDigits.eg(eg.test)
eg.test <- removeOuter(as.numeric(eg[-1]),0.5)
plotDigits.eg(eg.test)
eg.test <- removeOuter(as.numeric(eg[-1]),0.25)
plotDigits.eg(eg.test)
dev.off()

#绘制去噪之后各个数字的样本图形
test <- apply(plotData, 1, function(x){removeOuter(x, prob = 0.5)})
test <- t(test)
par(mfcol = c(10,10),mar = rep(0,4))
apply(as.matrix(test), 1, plotDigits.eg)
dev.off()

#应用于原始数据集去噪
dataset3 <- apply(raw_dataset[,-1], 1, function(x){removeOuter(x, prob = 0.5)})
dataset3 <- t(dataset3)
dataset3 <- as.data.frame(cbind(label = as.numeric(as.character(flag)), dataset3))

#施放部分内存
rm(raw_dataset)

#绘制去噪后的均值和方差图
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

#将dataset3转换成0-1矩阵，作为dataset4
index <- which(dataset3 > 0,arr.ind = T)
dataset4 <- dataset3
dataset4[index] <- 1

rm(eg.test,test,i,x,index)
rm(plotData)

#保存相应的数据集（注意修改相应的保存目录）
save.image("D:/R/DR/allDataset.RData")

rm(dataset2,dataset3,dataset4)

#奇异值分解
library(sp)
library(raster)
library(jpeg)

#找到分解后的特征
s <- proc.time()
model.set1.svd <- svd(dataset1)
print(s-proc.time())

#print(s-proc.time())
#用户    系统    流逝 
#-149.45   -0.81 -153.69

k <- seq(from = 50, to = 250, by = 50)
Kvect <- svd.setK(model.set1.svd,k,dataset1)

plot(Kvect)
dataset1.svd <- recovery(model.set1.svd,k = 250,dataset1)

save.image("D:/R/DR/set1_svd.RData")
