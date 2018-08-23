### Machine Learning Project###
###    Confidence Squared   ###
# 

library(dplyr)
library(data.table)
library(randomForest)
library(gbm)
setwd('~/Desktop/ML Project')
# Load Data

dtest <- read.csv("test.csv", stringsAsFactors = TRUE)
dtrain <- read.csv("train.csv", stringsAsFactors = TRUE)
testpx = read.csv("sample_submission.csv", stringsAsFactors = TRUE)
dtest = full_join(testpx,dtest)

#inner_join(superheroes, publishers)

# # Quick look at data
# summary(dtest)
# describe(dtest)
# summary(dtrain)
# describe(dtrain)
# # Plot missingness
# image(is.na(dtest), main = "Missing Values", xlab = "Observation", ylab = "Variable", 
#       xaxt = "n", yaxt = "n", bty = "n")
# axis(1, seq(0, 1, length.out = nrow(dtest)), 1:nrow(dtest), col = "white")
# axis(2, c(0, 0.5, 1), names(dtest), col = "white", las = 2)
# 
# image(is.na(dtrain), main = "Missing Values", xlab = "Observation", ylab = "Variable", 
#       xaxt = "n", yaxt = "n", bty = "n")
# axis(1, seq(0, 1, length.out = nrow(dtrain)), 1:nrow(dtrain), col = "white")
# axis(2, c(0, 0.5, 1), names(dtrain), col = "white", las = 2)


# Defining cleaning functions:

# add "None" as a level for group1
addLevel <- function(x){
  if(is.factor(x)) return(factor(x, levels=c(levels(x), "None")))
  return(x)
}

# For the rest columns, qualitative assign the most frequest, quantitative assign 0
missFun1 <- function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- 0
    #mean(x, na.rm = TRUE)
    x
  } else {
    x[is.na(x)] <- names(which.max(table(x)))
    x
  }
}

# Transform: fill missing and remove some columns
transFun <- function(t) {
  
  t <- subset(t, select=-c(MiscFeature,Fence,PoolQC,Alley,Street,Utilities,Condition2,RoofMatl,Id,PoolArea,LotFrontage))
  
  # missing GarageYrBlt should equal YearBuilt
  t$GarageYrBlt[is.na(t$GarageYrBlt)] = t$YearBuilt[is.na(t$GarageYrBlt)]
  
  # Separating groups for different functions
  cols = c("BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond")
  group1 <-t[,cols]
  group2 <-t[,!(colnames(t) %in% cols)]
  
  # adding "None" level
  group1 <- as.data.frame(lapply(group1, addLevel))
  group1[is.na(group1)] <- "None"
  
  # Assigning 0 or most regularly occuring value to variable
  group2=data.table(group2)
  group2[, lapply(.SD, missFun1)]
  
  # combine the two groups
  newt=cbind(group1,group2)
  
  
  return(newt)
  
}

# Process Test set and Training set in the same way
nTrain = transFun(dtrain)
nTest = transFun(dtest)

nTrain1 <- nTrain[complete.cases(nTrain), ]
nTrain1[is.na(nTrain1)]

nTest1 = nTest[complete.cases(nTest), ]
nTest1[is.na(nTest1)]


#fix data so bagging/random forests/boosting can be implemented
nTest1 <- rbind(nTrain1[1, ] , nTest1)
nTest1 <- nTest1[-1,]

# Implementing Random Forest formula to obtain bagging results
set.seed(1)
bag.housing <- randomForest(SalePrice ~ ., data = nTrain1, mtry = 69, importance = TRUE)
bag.housing

yhat.bag = predict(bag.housing, newdata = nTest1)
yhat.bag
housing.test = nTest1[,"SalePrice"]
housing.test
plot(yhat.bag, housing.test)
mean((yhat.bag - housing.test)^2)

importance(bag.housing)
varImpPlot(bag.housing)

#Varying the number of variables used at each step of the random forest procedure.
set.seed(0)
oob.err = numeric(69)
for (mtry in 1:69) {
  fit = randomForest(SalePrice ~ ., data = nTrain1, mtry = mtry)
  oob.err[mtry] = fit$mse[500]
  cat("We're performing iteration", mtry, "\n")
}

#Visualizing the OOB error.
plot(1:69, oob.err, pch = 16, type = "b",
     xlab = "Variables Considered at Each Split",
     ylab = "OOB Mean Squared Error",
     main = "Random Forest OOB Error Rates\nby # of Variables")


#Implementing Random Forest formula to obtain Random Forest results
set.seed(1)
housing.rf = randomForest(SalePrice ~ ., data = nTrain1, mtry = 23, importance = TRUE)
yhat.rf = predict(housing.rf, newdata = nTest1)
yhat.rf
housing.test = nTest1[,"SalePrice"]
housing.test
plot(yhat.rf, housing.test)
mean((yhat.rf - housing.test)^2)

importance(housing.rf)
varImpPlot(housing.rf)

#Implementing boosting formula to obtain boosting results
set.seed(1)
boost.housing = gbm(SalePrice~., data = nTrain1, distribution = "gaussian", n.trees=5000, interaction.depth=4)
summary(boost.housing)

#construct partial dependence plotof summarised variables
par(mfrow=c(1,2))

#graph most influential variables
plot(boost.housing, i = "X2ndFlrSF")
plot(boost.housing, i = "TotRmsAbvGrd")
plot(boost.housing, i = "GarageType")
plot(boost.housing, i = "GarageArea")
plot(boost.housing, i = "MasVnrArea")
plot(boost.housing, i = "ScreenPorch")
plot(boost.housing, i = "LotShape")
plot(boost.housing, i = "LotConfig")

#use boosted model to predict SalesPrice of test set
set.seed(1)
yhat.boost = predict(boost.housing, data = nTest1,
                      n.trees=5000)

mean((yhat.boost - housing.test)^2)





