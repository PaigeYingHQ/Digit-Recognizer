#加载数据
load("D:/R/DR/set1_svd.RData")

#PCA降维
dataset1.svd.cov <- cov(dataset1.svd)
model.set1.pca <- prcomp(dataset1.svd.cov)
summary(model.set1.pca)

#主成分分析结果
#cumulative proportion of variance
#dataset1: 58 factors can reach 99.5%
dataset1.svd.pca <- as.matrix(dataset1.svd) %*% model.set1.pca$rotation[,1:58]

train.flag <- flag[1:40000]
test.flag <- flag[40001:42000]

train <- dataset1.svd.pca[1:40000,]
test <- dataset1.svd.pca[40001:42000,]

#LDA
s <- proc.time()
lda.control <- trainControl(method = "cv",number = 10)
model.set1.svd.lda <- train(x = train,
                            y = train.flag,
                            method = "lda",
                            trControl = lda.control)
print(s-proc.time())
pred <- predict(model.set1.svd.lda,test)
accuracy(pred,test.flag)
#用户   系统   流逝 
#-20.08  -2.03 -22.17
#86.6%

#QDA
s <- proc.time()
qda.control <- trainControl(method = "cv",number = 10)
model.set1.svd.qda <- train(x = train,
                            y = train.flag,
                            method = "qda",
                            trControl = qda.control)
print(s-proc.time())
pred <- predict(model.set1.svd.qda,test)
accuracy(pred,test.flag)
#用户   系统   流逝 
#-10.37  -1.05 -11.44
#95.85%

#L2 Regularized Support Vector Machine (dual) with Linear Kernel
#高斯核
s <- proc.time()
model.set1.svd.ksvm <- ksvm(x = train,
                            y = train.flag,
                            kernel = "rbfdot",
                            C = 0.5,
                            cross = 3)
print(s-proc.time())
pred <- predict(model.set1.svd.ksvm,test)
accuracy(pred,test.flag)
#用户    系统    流逝 
#-757.12  -12.18 -770.08 
#96.6%

#线性核
s <- proc.time()
model.set1.svd.ksvm <- ksvm(x = train,
                          y = train.flag,
                          kernel = "vanilladot",
                          C = 0.5,
                          cross = 3)
print(s-proc.time())
pred <- predict(model.set1.svd.ksvm,test)
accuracy(pred,test.flag)
#用户    系统    流逝 
#-230.65   -3.21 -234.07
#92.6
#92.8%