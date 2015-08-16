library(caret)
library(AppliedPredictiveModeling)
library(pls)
library(e1071)
library(lattice)
library(pls)
library(MASS)
library(lars)
library(elasticnet)
library(car)
library(glmnet)

###read data
data<-read.csv("C:\\Users\\Nandakumar\\Documents\\GitHub\\Boston-house-pricing\\dataset.csv",stringsAsFactors=F,header=T)
head(data)
data<-data[,!names(data) %in% c('X','X.1')]

###do a scatterplot matrix to  see linearilty
splom(data)
nrow(data)
head(data)


###create training and test data set
###set 75 percent of rows for training and rest for test
bound<-floor(0.75*nrow(data))
data.train <- data[1:bound, ]            
data.test <- data[(bound+1):nrow(data), ] 
nrow(data.test)
nrow(data.train)
dataTrainX<-data.train
dataTestX<-data.test

###check skewness 
skewness(dataTrainX$RAD)
###apply box cox transformation
boxcox<-preProcess(dataTrainX,method ="BoxCox") 
dataTrainXtrans<-predict(boxcox,dataTrainX)
head(dataTrainXtrans)
hist(dataTrainXtrans$CRIM)
hist(dataTrainX$CRIM)

datatestXtrans<-predict(boxcox,dataTestX)
head(datatestXtrans)
hist(datatestXtrans$MEDV)
hist(dataTestX$MEDV)

###create training data
trainingData<-dataTrainXtrans
trainingData<-dataTrainX
head(trainingData)

###fit the model-OLS
model<-lm(MEDV~.,data=trainingData)
summary(model)
par(mfrow=c(2,2))
plot(model)

###predict values
pred<-predict(model,datatestXtrans)
###create obs,pred data frame
df<-data.frame(obs=datatestXtrans$MEDV,pred=pred)
df
defaultSummary(df)
###cross-validation
ctrl<-trainControl(method="cv",n=10)
set.seed(100)
tmp<-subset(dataTrainXtrans,select =-MEDV)
head(tmp)
modcv<-train(x=tmp,y=dataTrainXtrans$MEDV,method="lm",trControl =ctrl)


###check for multicollinearality
vif(model)
###vif levels shows collinearity in the dataset
###use ridge regression to check for estimate variation
ridge<-lm.ridge(MEDV~.,data=trainingData,lambda =seq(0,100,by=1))
plot(ridge)
summary(ridge)
###ridge estimates and plot suggest multicollinarity exists but it does not cause significant change in the 
###estimate values.To estimate parameters by reducing correlation, a PCA is done


###pca analysis 
pca<-data
###standardize independent variables
x<-subset(pca,select=-MEDV)
head(x)
x<-scale(x)
###center the dependent variable
y<-pca$MEDV
y<-scale(y,scale =F)
###do pca on indepenedent variables
comp<-princomp(x,cor =F,scores =T)
comp
plot(comp)
biplot(comp)
summary(comp)
###nine principal components explain 95% of the total variance
comp$scores
comp$loadings

pcr<-pcr(MEDV~.,data=trainingData,validation="CV")
summary(pcr)
###choose nine components for prediction
xpcr=subset(datatestXtrans,select=-MEDV)
pcrpred<-predict(pcr,xpcr,ncomp =9)
pcrdf<-data.frame(obs=datatestXtrans$MEDV,pred=pcrpred)
###find rmse
rmsepcr<-sqrt(mean((pcrdf$obs-pcrdf$MEDV.9.comps)^2))
###rmse is reduced to  0.087

###pls regression is a better variation of PCR.It accounts for the variation in response when selecting weights
###use pls package, plsr function
###default algorithm is Dayal and Mcgregor kernel algorithm
plsFit<-plsr(MEDV~.,data=trainingData,validation="CV")
###predict first five MEDV values using 1 and 2 components
pls.pred<-predict(plsFit,datatestXtrans[1:5,],ncomp=1:2)
summary(plsFit)
validationplot(plsFit,val.type ="RMSEP")
pls.RMSEP<-RMSEP(plsFit,estimate="CV")
plot(pls.RMSEP,main="RMSEP PLS",xlab="Components")
min<-which.min(pls.RMSEP$val)
points(min,min(pls.RMSEP$val),pch=1,col="red")
plot(plsFit, ncomp=9, asp=1, line=True)
###use 9 components
pls.pred2<-predict(plsFit,datatestXtrans,ncomp=9)
pls.eval<-data.frame(obs=datatestXtrans$MEDV,pred=pls.pred2[,1,1])
defaultSummary(pls.eval)

###generalized linear models





