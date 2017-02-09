#�Զ��庯������
#----------------------�Զ��庯��plotDigits()--------------------#
plotDigits <- function(x){
  x <- rev(x)
  m <- matrix(x,nrow = 28)
  m <- apply(m, 2, rev) 
  image(m,col = grey.colors(255))
}
#----------------------------------------------------------------#
#-------------------�Զ��庯��plotDigits.eg()--------------------#
plotDigits.eg <- function(x){
  x <- rev(x)
  m <- matrix(x,nrow = 28)
  m <- apply(m, 2, rev) 
  image(m,col = grey.colors(255),labels = F,tick = F)
}
#----------------------------------------------------------------#
#---------------�Զ��庯��plotDigits.output()--------------------#
plotDigits.output <- function(x,file){
  x <- rev(x)
  m <- matrix(x,nrow = 28)
  m <- apply(m, 2, rev)
  png(filename = file,bg = "transparent")
  print(image(m,col = grey.colors(255),labels = F,tick = F))
  dev.off()
}
#----------------------------------------------------------------#
#-----------------�Զ��庯��removeOuter()------------------------#
removeOuter <- function(x,prob = 0.25)
{
  num <- x[which(x != 0)]
  thres <- quantile(num,probs = 1-prob)
  index <- which(x < thres & x > 0)
  x[index] <- 0
  return(x)
}
#----------------------------------------------------------------#
#--------------------�Զ��庯��accuracy()------------------------#
accuracy <- function(pred,flag)
{
  return(length(which(pred == flag))/length(flag))
}
#----------------------------------------------------------------#
#--------------------�Զ��庯��recovery()------------------------#
recovery <- function(fac,k,dataset)
{
  dmat <- diag(k)
  diag(dmat) <- fac$d[1:k]
  m <- fac$u[,1:k] %*% dmat %*% t(fac$v[,1:k])
  return(m)
}
#----------------------------------------------------------------#
#--------------------�Զ��庯��svd.setK()------------------------#
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
#----------------------------------------------------------------#


