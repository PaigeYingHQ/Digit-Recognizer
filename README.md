&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;<big>数字识别（Digit Recognizer）
======
##<big>一、 内容概要</big>
###<big>1.数据集的介绍</big>
  <big>[MNIST](http://yann.lecun.com/exdb/mnist/)(THE MNIST DATABASE of handwritten digits)</big>
  

&#8195;&#8195;<big>此次使用的数据是从Kaggle的Digit Recoginzer这个比赛中下载的。并且，最后通过综合评价选出模型，将模型的预测结果进行了上传。</big>

&#8195;&#8195;<big>数据主要是由手写数字的__像素点__数据组成的，每一个像素点的取值为__[0,255]__之间，每个样本像素个数为28*28，即784个。以一个例子来具体说明，如下图1.1</big>

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;![example.png](http://upload-images.jianshu.io/upload_images/4154180-307e0554cbb6f17a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图1.1 样本像素图示例

<big>具体代码</big>
    
    plotDigits <- function(x){
      x <- rev(x)
      m <- matrix(x,nrow = 28)
      m <- apply(m, 2, rev) 
      image(m,col = grey.colors(255))
    }
    plotDigits(raw_dataset[10,])

###<big>2.内容介绍</big>
&#8195;&#8195;<big>给定训练集（train）其中包含42,000条样本信息，训练模型并用于预测另外28,000条样本的标识值（label）,其中每个手写数字占总样本的比例接近10%。</big>

##二、绘制样本图形
&#8195;&#8195;<big>通过绘制各个数字的样本图形来了解手写数字的一些特点，如下图2.1</big>

&#8195;&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-fd5476dedf8af354.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图2.1 各类数字样本示例图
&#8195;&#8195;

<big>具体代码</big>

    plotDigits.eg <- function(x){
      x <- rev(x)
      m <- matrix(x,nrow = 28)
      m <- apply(m, 2, rev) 
      image(m,col = grey.colors(255),labels = F,tick = F)
    }
    
    plotData <- matrix(ncol = NCOL(raw_dataset),nrow = 0)
    for(i in 0:9)
    {
      plotData <- rbind(plotData,subset(raw_dataset,label == i)[1:10,])
    }
    plotData <- plotData[,-1]
    par(mfcol = c(10,10),mar = rep(0,4))
    apply(plotData, 1, plotDigits.eg)
    dev.off()

&#8195;&#8195;<big>通过图2.1我们可以看到，由于是实际的手写数字，因此，并不像直接由电脑打印出来的文本数字，各个样本之间存在一定的差异性。但是各个类型的数字之间的区分或者界限还是比较清晰的，这从某种意义上也反映了这个问题的可行性和具体实现的难度。这一点也可以由下面的像素取均值和取方差后绘制的图形来说明，具体如下图2.2和图2.3</big>



&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-4e3ec4144abe6b09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图2.2 像素均值图



    par(mfrow = c(2,5),mar = rep(0.2,4))
    for(i in 0:9)
    {
      x <- apply(subset(raw_dataset,label == i )[,-1],2,mean)
      plotDigits.eg(x)
    }
    dev.off()

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-eee7f32ae6763e90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图2.3 像素方差图

    par(mfrow = c(2,5),mar = rep(0.2,4))
    for(i in 0:9)
    {
      x <- apply(subset(raw_dataset,label == i)[,-1],2,sd)
      plotDigits.eg(x)
    }
    dev.off()

&#8195;&#8195;<big>从像素方差图可以看到，各个类别的数字内部的差异性，同时各个数字核心笔画的周围都存在的一定的模糊，这也表示实际手写数字中在这些模糊部分也会出现，只是占总体比例的多少问题，如何正确识别模糊部分的信息也是我们需要考虑的问题。</big>

&#8195;&#8195;__<big>通过前两部分的阐述，我们可以对此次的数据进行如下几点的概括：
</big>__

&#8195;&#8195;&#8195;&#8195;<big>1.由于图片本身的特点导致的数据的维度较高和数据量较大（40,000条样本*784个像素点）</big>

&#8195;&#8195;&#8195;&#8195;<big>2.数据集是由灰度图片得到的，因此存在一定比例的0，此时数据集也可近似看作一个稀疏矩阵</big>

&#8195;&#8195;&#8195;&#8195;<big>3.手写数字相对与规范的输出有更多随机的部分，例如7中间可能添加一条斜线来标示它具体是7还是1</big>

&#8195;&#8195;&#8195;&#8195;<big>4.部分手写数字1可能出现倾斜</big>

&#8195;&#8195;__<big>通过上面的小结我们在选择模型时需要注意以下的问题：</big>__

&#8195;&#8195;&#8195;&#8195;<big>1.选择什么样的模型来实现更高效率的计算和预测？</big>

&#8195;&#8195;&#8195;&#8195;<big>2.是否需要对像素图进行去噪？去噪的比例如何确定？</big>

&#8195;&#8195;&#8195;&#8195;<big>3.若考虑通过降低维度来增加模型的选择范围，那么对于图像数据应该选择怎样的降维方式？</big>

##<big>三、神经网络模型的建立</big>
&#8195;&#8195;&#8195;<big>在不考虑降维的情况下，直接使用原始数据进行建模需要消耗大量的时间，尤其是对于哪些计算复杂度在o(n^2)以上的算法。但是，我们可以尝试在一些开源引擎上运行分布式算法（parallel distributed machine learning algorithms），例如广义线性模型（generalized linear models）、随机森林（Random Forest）和集群环境下的神经网络模型（neural networks (deep learning) within cluster environments）。</big>

&#8195;&#8195;&#8195;<big>由于是图形识别问题，首选应该是卷积神经网络模型（Convolutional Neural Network,CNN ），但是R中没有CNN的接口，因此选择基于[H2O](http://www.h2o.ai/)环境下使用标准的二层神经网络模型进行建模，每层有100个结点，0.5的隐含层结点不参与工作（dropout ratio）。模型的代码如下所示：</big>

    #required libs
    library(statmod)
    library(methods)
    library(h2o)
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
    #load datasets
    load("D:/R/DR/allDataset.RData")
    rm(dataset2,dataset3,dataset4)
    
    dataset1 <- cbind(label = flag,dataset1)
    train1 <- dataset1[1:40000,]
    test1 <- dataset1[40001:42000,]
    set1.dl.pred <- deeplearning_h2o(train1,test1)
    accuracy(set1.dl.pred$Label,flag[40001:42000])

&#8195;&#8195;<big>输出结果: </big>

&#8195;&#8195;<big>预测的正确率为94.5%,下表所示为10折交叉验证的error和accuracy（error = bias + variance + noise)，如下表3.1</big>

&#8195;&#8195;&#8195;&#8195;表3.1 十折交叉验证正确率和错误值汇总表
	  <table>
   		  <tr>
      		  <td><big>cv_valid</big></td>
      		  <td>1</td>
      		  <td>2</td>
      		  <td>3</td>
      		  <td>4</td>
      		  <td>5</td>
   		  </tr>
   		  <tr>
      		  <td><big>accuracy</big></td>
      		  <td>0.9422</td>
      		  <td>0.9465</td>
      		  <td>0.9429</td>
      		  <td>0.9453</td>
      		  <td>0.9423</td>
   		  </tr>
   		  <tr>
      		  <td><big>err</big></td>
      		  <td>0.0578</td>
      		  <td>0.0535</td>
      		  <td>0.05708</td>
      		  <td>0.05469</td>
      		  <td>0.05768</td>
   		  </tr>
   		  <tr>
      		  <td><big>cv_valid</big></td>
      		  <td>6</td>
      		  <td>7</td>
      		  <td>8</td>
      		  <td>9</td>
      		  <td>10</td>
   		  </tr>
   		  <tr>
      		  <td><big>accuracy</big></td>
      		  <td>0.9455</td>
      		  <td>0.9458</td>
      		  <td>0.9405</td>
      		  <td>0.9494</td>
      		  <td>0.9346</td>
   		  </tr>
   		  <tr>
      		  <td>err</big></td>
      		  <td>0.05449</td>
      		  <td>0.05411</td>
      		  <td>0.05942</td>
      		  <td>0.0505</td>
      		  <td>0.0653</td>
   		  </tr>
	  </table>

&#8195;&#8195;<big>最后得到的模型的预测正确率为94.5%，error的均值为0.05647。由于error等于偏差、方差和噪声的和，因此error越小代表模型的预测能力越强。与此同时，我们也可以看到每一折的error都大致在均值附近随机波动，这也说明了这个模型对于预测这个数据集的稳定性较强，即在出现极其偏离训练样本之前，模型对于这类数据具有一定的预测能力，可以在这个模型上对算法进行进一步的优化来扩展它对于其他类型具有偏差数据的预测能力。</big>
##<big>四、线性模型的建立</big>
&#8195;&#8195;<big>神经网络模型并未考虑降维，所以可以尝试一些其他的算法进行比较。由于训练集可以看作一个大的稀疏矩阵，同时各类数字内的方差远大于各类数字之间的方差，综合以上两点接下来主要考虑的是先进行数据的去噪和降维，再使用线性模型进行建模。</big>
###<big>4.1 剔除噪音</big>
&#8195;&#8195;<big>由于像素信息具有一定的冗余，因此首先考虑的是剔除部分噪声，主要使用了两种方法，它们分别是简单过滤法和奇异值分解。</big>
####<big>4.1.1 剔除多余像素</big>
&#8195;&#8195;<big>首先需要考虑的是保留多少的像素信息，因此我通过一些具体的例子来判断并确定这个比例，如下图4.1，从左到右依次是保留100%、75%、50%和25%：</big>

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-4a777188334aef3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    
&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图4.1 剔除像素信息示例图

    removeOuter <- function(x,prob = 0.25)
    {
      num <- x[which(x != 0)]
      thres <- quantile(num,probs = 1-prob)
      index <- which(x < thres & x > 0)
      x[index] <- 0
      return(x)
    }
    par(mfrow = c(1,4),mar = rep(0.2,4))
    plotDigits.eg(as.numeric(eg[-1]))
    eg.test <- removeOuter(as.numeric(eg[-1]),0.75)
    plotDigits.eg(eg.test)
    eg.test <- removeOuter(as.numeric(eg[-1]),0.5)
    plotDigits.eg(eg.test)
    eg.test <- removeOuter(as.numeric(eg[-1]),0.25)
    plotDigits.eg(eg.test)
    dev.off()

&#8195;&#8195;<big>当只保留25%像素信息时数字可能出现缺损，但又要剔除尽可能多的信息，因此综合考虑选择保留50%的像素信息，并对所有的样本进行相同的处理。对于处理后的样本信息进行输出得到如下图4.2所示：</big>

&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-cca7b2d00d31bbb8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图4.2 剔除像素信息后各数字样本示例图

&#8195;&#8195;<big>通过比较去噪前后的样本方差图来考察去噪的成效，比较去噪前后各类数字的分布是否变更加清晰，如下图4.3所示：</big>

&#8195;&#8195;&#8195;&#8195;&#8195;![image.png](http://upload-images.jianshu.io/upload_images/4154180-fdece51b9e8cbbe5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;图4.3 去噪前后方差对比图

&#8195;&#8195;<big>可以看到很多数字的棱角开始变得清晰，但是仍然存在一定的模糊，这从另一个方面证明手写数字具有一定范围内“随机”的特点。因此，仅仅通过去噪所带来的对于问题的简化其成效是有限的。</big>
###<big>4.2 奇异值分解</big>
&#8195;&#8195;<big>
在简单过滤的基础进行奇异值分解进一步地剔除噪声。相对于特征值分解，奇异值分解可以应用一般的矩阵，可以应用于减噪和降维。其几何意义就是将原有的数据向多个新的正交基进行线性变换，并保留包含多数信息的矩阵来近似原有的矩阵。
</big>

&#8195;&#8195;<big>
这里需要注意的是，一般地，矩阵之间近似程度可以使用Frobenius范数来具体衡量，其中矩阵的Frobenius范数是矩阵所有元素平方和的平方。:
</big>


    svd.setK <- function(fac,k,dataset)
    {
      re <- c()
      n <- NCOL(dataset)*NROW(dataset)
      for(i in k) {
    dmat <- diag(i)
    diag(dmat) <- fac$d[1:i]
    m <- fac$u[,1:i] %*% dmat %*% t(fac$v[,1:i])
    re <- c(re,sum(abs(dataset-m))/n)
      }
      return(re)
    }
    
    #required libs
    library(sp)
    library(raster)
    library(jpeg)
    
    #SVD
    s <- proc.time()
    model.set1.svd <- svd(dataset1)
    print(s-proc.time())
    k <- seq(from = 50, to = 250, by = 50)
    Kvect <- svd.setK(model.set1.svd,k,dataset1)
    
    plot(Kvect)
    dataset1.svd <- recovery(model.set1.svd,k = 250,dataset1)


&#8195;&#8195;<big>
当其为0时，两个矩阵严格相等。但是，由于在选择保留多少信息时需要确定K值，K越大，则估计矩阵更接近原矩阵。然而，在SVD算法下仅通过Frobenius范数来确定K，K倾向于尽可能取大，当K取太大时则失去了减噪的意义。这也意味着我们需要使用其他的指标来确定K值。这里选择的是当两个矩阵单位像素误差减少速度明显减慢时，取该K值作为最后的K取值。
</big>

###<big>4.3 对去噪后的数据进行降维</big>

&#8195;&#8195;<big>由于数据的维度较高，有监督的降维需要消耗大量的时间，因此先尝试使用无监督降维中的主成分分析（Principal Component Analysis, PCA）降维。PCA的主要思想是用较少的综合变量来代替原来较多的变量，同时要求这几个综合变量尽可能多地反映原来变量的信息，并且彼此之间互不相关。</big>

    #PCA
    dataset1.cov <- cov(dataset1)
    model.set1.pca <- prcomp(dataset1.cov)
    summary(model.set1.pca)
    
    #cumulative proportion of variance
    #dataset1: 58 factors can reach 99.5%
    dataset1.pca <- as.matrix(dataset1) %*% model.set1.pca$rotation[,1:58]

###<big>4.4 再次建立模型</big>
&#8195;&#8195;<big>考虑到数据类内方差远小于类间方差，因此考虑使用计算时间相对较少的线性模型。本次使用的模型有线性判别分析、二次判别分析和非线性支持向量机。其中支持向量机尝试了两种核：高斯核和线性核。这里以其中一个处理过的数据集为例来展示代码</big>

    #required libs
    library(lattice)
    library(ggplot2)
    library(caret)
    library(MASS)
    library(kernlab)
    
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

    #L2 Regularized Support Vector Machine (dual) with Linear Kernel
    #Gaussian kernel
    s <- proc.time()
    model.set1.ksvm <- ksvm(x = train,
    						y = train.flag,
    						kernel = "rbfdot",
    						C = 0.5,
    						cross = 3)
    print(s-proc.time())
    pred <- predict(model.set1.ksvm,test)
    accuracy(pred,test.flag)
    
    #linear kernel
    s <- proc.time()
    model.set1.ksvm <- ksvm(x = train,
    						y = train.flag,
    						kernel = "vanilladot",
    						C = 0.5,
    						cross = 3)
    print(s-proc.time())
    pred <- predict(model.set1.ksvm,test)
    accuracy(pred,test.flag)

###<big>4. 模型的评价</big>

&#8195;&#8195;<big>
对于模型的评价主要从以下两个方面来考虑。首先是预测的正确率，其次是建立模型耗费的时间。具体如下表5.1:
</big>

&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;<big>
表5.1 各模型表现指标汇总表</big>
<div align = "center">
<table>
     <tr>
          <td><big><b>模型</b></big></td>
          <td>2layers-NN</td>
          <td>LDA</td>
          <td>QDA</td>
          <td>KSVM-Gaussian</td>
          <td>KSVM-linear</td>
     </tr>
     <tr>
          <td><big><b>正确率</b></big></td>
          <td>94.50%</td>
          <td>86.60%</td>
          <td>95.85%</td>
          <td>96.60%</td>
          <td>92.80%</td>
     </tr>
     <tr>
          <td><big><b>运算时间(s)</b></big></td>
          <td>240</td>
          <td>22.17</td>
          <td>11.14</td>
          <td>770.08</td>
          <td>234.07</td>
     </tr>
     <tr>
          <td><big><b>特征个数</b></big></td>
          <td>784</td>
          <td>58</td>
          <td>58</td>
          <td>58</td>
          <td>58</td>
     </tr>
     <tr>
          <td><big><b>单位特征的运算时间</b></big></td>
          <td>0.306122449</td>
          <td>0.382241379</td>
          <td>0.192068966</td>
          <td>13.27724138</td>
          <td>4.035689655</td>
     </tr>
     <tr>
          <td><big><b>原始数据运算时间比较</b></big></td>
          <td>0.00%</td>
          <td>9.80%</td>
          <td>18.39%</td>
          <td>17.20%</td>
          <td>-1.63%</td>
     </tr>
</table>
</div>

&#8195;&#8195;<big>综表5.1 最后一行表示数据在未进经减噪时直接进行降维和建模所多消耗的时间百分比。
综合考虑PCA-QDA是较为理想的方法。虽然PCA-KSVM（Gaussian）正确率比PCA-QDA高0.75%，但是其运算时间较长，且随着特征数量的增加，其运算时间呈幂级增长。神经网络是目前计算机视觉方向较为流行的算法，其具有的潜力较大。在此基础上进一步增加样本数量，其预测的正确率可以达到96%及以上，并且其单位特征的运算时间远低于KSVM算法。
</big>

&#8195;&#8195;<big>最后是对这次数字识别实践的一点总结。首先是去噪问题。通过具体的操作发现，去噪本身这个问题可以做非常大的拓展，比如寻找在处理图像时的最佳去噪方式和去噪比例，使算法提高运算速度、预测准确度，同时又不抹去过多的信息导致模型的泛化能力下降。其次是降维，特别是对图像数据，应尽可能地选择特定方法去提取数据的内在结果，而不是基于当下的部分信息去对原始数据做过多的处理。
</big>
