#lib
library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(kernlab)

#加载数据
load("D:/R/DR/allDataset.RData")
rm(dataset2,dataset3,dataset4)

#PCA降维
dataset1.cov <- cov(dataset1)
model.set1.pca <- prcomp(dataset1.cov)
summary(model.set1.pca)

#主成分分析结果
#cumulative proportion of variance
#dataset1: 58 factors can reach 99.5%
dataset1.pca <- as.matrix(dataset1) %*% model.set1.pca$rotation[,1:58]

train.flag <- flag[1:40000]
test.flag <- flag[40001:42000]

train <- dataset1.pca[1:40000,]
test <- dataset1.pca[40001:42000,]

#LDA
s <- proc.time()
lda.control <- trainControl(method = "cv",number = 10)
model.set1.lda <- train(x = train,
                        y = train.flag,
                        method = "lda",
                        trControl = lda.control)
print(s-proc.time())
pred <- predict(model.set1.lda,test)
accuracy(pred,test.flag)

#建模时间和预测的正确率
#用户   系统   流逝 
#-23.25  -1.25 -24.58 
#86.55%

#QDA
s <- proc.time()
qda.control <- trainControl(method = "cv",number = 10)
model.set1.qda <- train(x = train,
                        y = train.flag,
                        method = "qda",
                        trControl = qda.control)
print(s-proc.time())
pred <- predict(model.set1.lda,test)
accuracy(pred,test.flag)

#建模时间和预测的正确率
#用户   系统   流逝 
#-12.53  -1.09 -13.65
#95.8%

#L2 Regularized Support Vector Machine (dual) with Linear Kernel
#高斯核
s <- proc.time()
model.set1.ksvm <- ksvm(x = train,
                        y = train.flag,
                        kernel = "rbfdot",
                        C = 0.5,
                        cross = 3)
print(s-proc.time())
pred <- predict(model.set1.ksvm,test)
accuracy(pred,test.flag)
#用户    系统    流逝 
#-902.39  -26.02 -930.02 
#96.6%

#线性核
s <- proc.time()
model.set1.ksvm <- ksvm(x = train,
                        y = train.flag,
                        kernel = "vanilladot",
                        C = 0.5,
                        cross = 3)
print(s-proc.time())
pred <- predict(model.set1.ksvm,test)
accuracy(pred,test.flag)
#用户    系统    流逝 
#-227.66   -2.46 -230.32
#92.8%

#加载数据
load("D:/R/DR/allDataset.RData")
rm(dataset1,dataset2,dataset3)

#PCA
dataset4 <- dataset4[,-1]
dataset4.cov <- cov(dataset4)
model.set4.pca <- prcomp(dataset4.cov)
#dataset4: 156 factors can reach 99.5%
dataset4.pca <- as.matrix(dataset4) %*% model.set4.pca$rotation[,1:156]

train.flag <- flag[1:40000]
test.flag <- flag[40001:42000]

train <- dataset4.pca[1:40000,]
test <- dataset4.pca[40001:42000,]

#LDA
s <- proc.time()
lda.control <- trainControl(method = "cv",number = 10)
model.set4.lda <- train(x = train,
                        y = train.flag,
                        method = "lda",
                        trControl = lda.control)
print(s-proc.time())
pred <- predict(model.set4.lda,test)
accuracy(pred,test.flag)
#用户    系统    流逝 
#-141.69   -6.16 -148.31 
#85.5%

#QDA
s <- proc.time()
qda.control <- trainControl(method = "cv",number = 10)
model.set4.qda <- train(x = train,
                        y = train.flag,
                        method = "qda",
                        trControl = qda.control)
print(s-proc.time())
pred <- predict(model.set4.qda,test)
accuracy(pred,test.flag)
#用户   系统   流逝 
#-55.72  -4.20 -60.05 
#93.05%

#高斯核SVM
s <- proc.time()
model.set4.ksvm <- ksvm(x = train,
                        y = train.flag,
                        kernel = "rbfdot",
                        C = 0.5,
                        cross = 3)
print(s-proc.time())
pred <- predict(model.set4.ksvm,test)
accuracy(pred,test.flag)
#用户     系统     流逝 
#-2893.69   -91.76 -2990.46 
#94.4%

#线性核SVM
s <- proc.time()
model.set4.ksvm <- ksvm(x = train,
                        y = train.flag,
                        kernel = "vanilladot",
                        C = 0.5,
                        cross = 3)
print(s-proc.time())
pred <- predict(model.set4.ksvm,test)
accuracy(pred,test.flag)
#用户     系统     流逝 
#-1037.14   -10.07 -1048.44
#91.85%

