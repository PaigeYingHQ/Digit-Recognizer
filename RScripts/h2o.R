#Deeplearning applied to all datasets
library(statmod)
library(methods)
library(h2o)
#--------------------自定义函数deeplearning()---------------------------#
deeplearning_h2o <- function(train,test)
{
  
  localH2O = h2o.init(max_mem_size = '2g',
                      nthreads = -1)
  
  train_h2o <- as.h2o(train)
  test_h2o <- as.h2o(test)
  
  s <- proc.time()
  
  model <- h2o.deeplearning(x = 2:785,
                            y = 1,
                            training_frame = train_h2o,
                            activation = "RectifierWithDropout",
                            input_dropout_ratio = 0.2,
                            hidden_dropout_ratios = c(0.5,0.5),
                            balance_classes = TRUE,
                            hidden = c(100,100),
                            momentum_stable = 0.99,
                            nesterov_accelerated_gradient = TRUE,
                            epochs = 15,
                            nfolds = 10,
                            keep_cross_validation_predictions = TRUE,
                            keep_cross_validation_fold_assignment = TRUE)
  print(s - proc.time())
  h2o.confusionMatrix(model)
  h2o_y_test <- h2o.predict(model,test_h2o)
  
  df_y_test = as.data.frame(h2o_y_test)
  df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)),
                         Label = df_y_test$predict)
  return(df_y_test)
}
#------------------------------------------------------------------------#
#加载数据集
load("D:/R/DR/allDataset.RData")
rm(dataset2,dataset3,dataset4)

#dataset1
dataset1 <- cbind(label = flag,dataset1)
train1 <- dataset1[1:40000,]
test1 <- dataset1[40001:42000,]
set1.dl.pred <- deeplearning_h2o(train1,test1)
accuracy(set1.dl.pred$Label,flag[40001:42000])
#94.5%

#user  system elapsed 
#-1.72   -0.14 -240.44

#err
#[1] "0.05286783"  "0.05116164"  "0.05441504"  "0.05629222"  "0.050616972" "0.049469966"
#[7] "0.054808423" "0.052056883" "0.053283766" "0.05052995" 
#mean err: 0.05255027


#释放空间
rm(dataset1,set1.dl.pred)
rm(train1,test1)

#加载数据集
load("D:/R/DR/allDataset.RData")
rm(dataset1,dataset3,dataset4)

#dataset2
dataset2 <- cbind(label = flag,dataset2)
train2 <- dataset2[1:40000,]
test2 <- dataset2[40001:42000,]
set2.dl.pred <- deeplearning_h2o(train2,test2)
accuracy(set2.dl.pred$Label,flag[40001:42000])
#93.8%

#user  system elapsed 
#-2.08   -0.11 -230.32

#释放空间
rm(dataset2,set2.dl.pred)
rm(train2,test2)
