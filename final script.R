#lib
library(lattice)
library(ggplot2)
library(caret)
library(MASS)

#读入train
raw_dataset <- read.csv("digitRecg_train.csv",header = T,stringsAsFactors = F)

#分别提取flag和train
flag <- as.factor(raw_dataset[,1])
train <- raw_dataset[,-1]

#释放空间
rm(raw_dataset)

#读入test
test <- read.csv("digitRecg_test.csv",header = T,stringsAsFactors = F)

#PCA降维
train.cov <- cov(train)
model.pca <- prcomp(train.cov)
summary(model.pca)
train.pca <- as.matrix(train) %*% model.pca$rotation[,1:58]
test.pca <- as.matrix(test) %*% model.pca$rotation[,1:58]

#QDA
qda.control <- trainControl(method = "cv",number = 10)
model.qda <- train(x = train.pca,
                   y = flag,
                   method = "qda",
                   trControl = qda.control)
pred <- predict(model.qda,test.pca)
write.csv(pred,"submission.csv",row.names = F)
