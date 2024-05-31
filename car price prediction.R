##library
library("corrplot")
library("car")
library("moments")        
library("MASS")           
library("leaps")         
library("corrplot")       
library("randomForest")   
library("ggplot2")
library("glmnet")
library("e1071")

###read data----
car = read.csv("ford.csv", header = T)
dim(car)
head(car)

### Data cleaning-----

#remove names 
car = car[,-1]
head(car)
summary(car)

#fuel type category has 5 categories, 99% of data in two of them, so we remove other three category 
table(car$fuelType)#other category is one/Hybrid is 22/ Electric is 2

which(car$fuelType == "Other")
car = car[-which(car$fuelType == "Other"),]

which(car$fuelType == "Hybrid")
car = car[-which(car$fuelType == "Hybrid"),]

which(car$fuelType == "Electric")
car = car[-which(car$fuelType == "Electric"),]


table(car$transmission)#ok

#remove car that year is 2060
car = car[-which(car$year > 2021),]

#remove car that engineSize = 0
car = car[-which(car$engineSize < 1),]

#change categorical variable to factor
for (i in c(3,5)) {
  car[,i] = as.factor(car[,i])
  
}

str(car)

###Visual analysis for numeric variables-----

#scatter plot-----
par(mfrow = c(2, 3))  
for (i in c(1,4,6,7,8)) {
  plot(x = car[,i], y = car$price, main = paste(colnames(car)[i]), cex.lab=2, cex.main=3)
}
par(mfrow = c(1, 1))

#box-plot-----
ggplot()+
  geom_boxplot(data = car, aes(car$transmission, car$price, color = car$transmission, fill = car$transmission),alpha = 0.5)+
  xlab("Transmission") + ylab("Price")+
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  theme_classic(base_size = 20)

ggplot()+
  geom_boxplot(data = car, aes(car$fuelType, car$price, color = car$fuelType, fill = car$fuelType),alpha = 0.5)+
  xlab("Fuel Type") + ylab("Price")+
  scale_color_manual(values=c("#999999", "#E69F00"))+
  scale_fill_manual(values=c("#999999", "#E69F00"))+
  theme_classic(base_size = 20)

#correlation table----- 
cor_table <- round(cor(car[, c(1,2,4,6,7,8)]), 2)
corrplot(cor_table)










###Divide Data set into Train and Test----
set.seed(1122)
train_ind = sample(1:nrow(car), nrow(car) * 0.8)
train= car[train_ind,]
test = car[- train_ind,]

###Model 1: (linear regression)-------
lm_1 = lm(price~., data = train)
summary(lm_1)
par(mfrow = c(2, 2))
plot(lm_1)
par(mfrow = c(1, 1))

###Model 2: (transformed data)-----
##Box-Cox Transformation
box_results <- boxcox(price ~ ., data = train, lambda = seq(-5, 5, 0.1))               
box_results <- data.frame(box_results$x, box_results$y)            
lambda <- box_results[which(box_results$box_results.y == max(box_results$box_results.y)), 1]
lambda #0.2
train$T_price = (((train$price)^0.2)-1)/0.2

#Hist1
ggplot(data = train, aes(train$price))+
  geom_histogram(data = train, bins = 15)+
  theme_classic(base_size = 20)
#Hist2
ggplot(data = train, aes(train$T_price))+
  geom_histogram(data = train, bins = 15)+
  theme_classic(base_size = 20)

lm_2 = lm(T_price~.  -price , data = train)
summary(lm_2)
par(mfrow = c(2, 2))
plot(lm_2)
par(mfrow = c(1, 1))

##Model 3: (Best Subset Selection)-----
set.seed(1122)
bestsub_1 <- regsubsets(T_price~.-price, nvmax = 8, data = train, method = "exhaustive")
summary(bestsub_1)

#R-squared
summary(bestsub_1)$adjr2
#Plot Adjusted R-squared
plot(summary(bestsub_1)$adjr2,
     type = "b",
     xlab = "# of Variables", 
     ylab = "AdjR2", 
     xaxt = 'n',
     xlim = c(1, 8)); grid()
axis(1, at = 1: 8, labels = 1: 8)

points(which.max(summary(bestsub_1)$adjr2), 
       summary(bestsub_1)$adjr2[which.max(summary(bestsub_1)$adjr2)],
       col = "red", cex = 2, pch = 20)

coef(bestsub_1, 7)

#best lm---
lm_3 = lm(T_price~.-price ,data = train)
summary(lm_3)

##Test the Model-----
#Prediction

test$T_price = (((test$price)^0.2)-1)/0.2
pred_linear_reg <- predict(lm_3, test)
pred_lm_3 = (0.2*pred_linear_reg + 1)^5

#Create a dataframe to save prediction results on test data set
models_comp <- data.frame("Mean of AbsErrors"   = mean(abs(pred_lm_3 - test$price)),
                          "Mean of AbsErrors" = median(abs(pred_lm_3 - test$price)),
                          "Min of AbsErrors" = min(abs(pred_lm_3 - test$price)),
                          "Max of AbsErrors" = max(abs(pred_lm_3 - test$price)),
                          "Accuracy" =  mean(1-(abs(pred_lm_3 - test$price)/test$price))*100,                      
                          row.names = 'best linear regression')
View(models_comp)

#Actual vs. Predicted
plot(test$price, pred_lm_3, main = 'best linear regression',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)
abline(a = 0, b = 1, col = "red", lwd = 2)

###Model 4: Ridge Regression------------------------
x <- model.matrix(T_price~.-price,data = train)[, -1] #remove intercept
y <- train$T_price

lambda_grid <- 10 ^ seq(5, -2, length = 100)
lambda_grid

#Apply Ridge Regression (alpha = 0)
ridgereg_1 <- glmnet(x, y, alpha = 0, lambda = lambda_grid, standardize = TRUE, intercept = TRUE)
dim(coef(ridgereg_1))

#Plot Reg. Coefficients vs. Log Lambda
plot(ridgereg_1, xvar = "lambda")

#Cross validation to choose the best model
set.seed(1234)
ridge_cv    <- cv.glmnet(x, y, alpha = 0, lambda = lambda_grid, nfolds = 10)
#The mean cross-validated error
ridge_cv$cvm
#Estimate of standard error of cvm.
ridge_cv$cvsd

#value of lambda that gives minimum cvm
ridge_cv$lambda.min

#Coefficients of regression w/ best_lambda
ridgereg_2 <- glmnet(x, y, alpha = 0, lambda = ridge_cv$lambda.min, standardize = TRUE, intercept = TRUE)
coef(ridgereg_2)

##Test the Model----------------------------------
#Model: ridgereg_2
#Prediction
#Create model matrix for test
x_test <- model.matrix(T_price~.-price,data = test)[, -1]#remove intercept
pred_ridgereg <- predict(ridgereg_2, s = ridge_cv$lambda.min, newx = x_test)
pred_ridgereg
pred_ridgereg  <- (pred_ridgereg*0.2+1)^5
pred_ridgereg
#Absolute error mean, median, sd, max, min


models_comp <- rbind(models_comp, "RidgeReg" = c(mean(abs(pred_ridgereg - test$price)),
                                                 median(abs(pred_ridgereg - test$price)),
                                                 min(abs(pred_ridgereg - test$price)),
                                                 max(abs(pred_ridgereg - test$price)),
                                                 mean(1-(abs(pred_ridgereg - test$price)/test$price))*100))


View(models_comp)

#Actual vs. Predicted
plot(test$price, pred_ridgereg, main = 'RidgeReg',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)

abline(a = 0, b = 1, col = "red", lwd = 2)












###Model 5: Lasso Regression------------------------

#Apply LASSO Regression (alpha = 1)
lassoreg_1 <- glmnet(x, y, alpha = 1, lambda = lambda_grid, standardize = TRUE, intercept = TRUE)
dim(coef(lassoreg_1))

#Plot Reg. Coefficients vs. Log Lambda
plot(lassoreg_1, xvar = "lambda")

#Cross validation to choose the best model
set.seed(1234)
lasso_cv    = cv.glmnet(x, y, alpha = 1, lambda = lambda_grid, nfolds = 10)
#The mean cross-validated error
lasso_cv$cvm
#Estimate of standard error of cvm.
lasso_cv$cvsd

#value of lambda that gives minimum cvm
lasso_cv$lambda.min

#Coefficients of regression w/ best_lambda
lassoreg_2 <- glmnet(x, y, alpha = 1, lambda = lasso_cv$lambda.min, standardize = TRUE, intercept = TRUE)
coef(lassoreg_2)

##Test the Model----------------------------------
#Model: lassoreg_2
#Prediction
pred_lassoreg <- predict(lassoreg_2, s = lasso_cv$lambda.min, newx = x_test)
pred_lassoreg
pred_lassoreg <- (pred_lassoreg*0.2+1)^5
pred_lassoreg
#Absolute error mean, median, sd, max, min

models_comp <- rbind(models_comp, "lassoreg" = c(mean(abs(pred_lassoreg - test$price)),
                                                 median(abs(pred_lassoreg - test$price)),
                                                 min(abs(pred_lassoreg - test$price)),
                                                 max(abs(pred_lassoreg - test$price)),
                                                 mean(1-(abs(pred_lassoreg - test$price)/test$price))*100))


View(models_comp)

#Actual vs. Predicted
plot(test$price, pred_lassoreg, main = 'LASSOReg',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)
abline(a = 0, b = 1, col = "red", lwd = 2)

###model 6: Random Forest------------
#cheak variable importance
set.seed(1122)
rf_1 <- randomForest(T_price~.-price, data = train, 
                     mtry = 2, ntree = 500, nodesize = 5, importance = TRUE)
varImpPlot(rf_1)


#feature selection
set.seed(1122)
rf_cv_1 = rfcv(trainx =  train[, c(-2,-9)],
               trainy = train$T_price,
               cv.fold = 10,
               step = 0.9,
               mtry = function(p) max(1, floor(p/3)), 
               recursive = FALSE)
str(rf_cv_1)
rf_cv_1$error.cv
rf_cv_1$predicted
which.min(rf_cv_1$error.cv) #6
sort(importance(rf_1)[,1])

plot(rev(rf_cv_1$error.cv),
     type = "b",
     xlab = "number of Variables", 
     ylab = "error", 
     xaxt = 'n',
     xlim = c(1, 7)); grid()
axis(1, at = 1:7, labels = 1:7)

points(which.min(rev(rf_cv_1$error.cv)), 
       rev(rf_cv_1$error.cv)[which.min(rev(rf_cv_1$error.cv))],
       col = "red", cex = 2, pch = 20)

rf_4 <- randomForest(T_price ~ year  + mpg  + engineSize + mileage + transmission +
                       tax  , data = train, 
                     mtry = 2, ntree = 500, nodesize = 5)
min(rf_4$mse)
##Test the Model----------------------------------
#Prediction: rf_4
pred_rf  = predict(rf_4, test)
pred_rf  = (0.2*pred_rf + 1)^5
pred_rf

#Absolute error mean, median, sd, max, min

models_comp <- rbind(models_comp, "RandomForest" = c(mean(abs(pred_rf - test$price)),
                                                     median(abs(pred_rf - test$price)),
                                                     min(abs(pred_rf - test$price)),
                                                     max(abs(pred_rf - test$price)),
                                                     mean(1-(abs(pred_rf - test$price)/test$price))*100))


View(models_comp)

#Actual vs. Predicted
plot(test$price, pred_rf, main = 'RandomForest',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)
abline(a = 0, b = 1, col = "red", lwd = 2)

###Model 7: Bagging--------------------------------
set.seed(11)
bagging_1 <- randomForest(T_price~.-price, data = train, mtry = ncol(train) - 2, ntree = 500)
bagging_1

##Test the Model----------------------------------
#Prediction: M8 Bagging
pred_bagging  <- predict(bagging_1, test)
pred_bagging  <- (pred_bagging*0.2+1)^5
pred_bagging

#Absolute error mean, median, sd, max, min

models_comp <- rbind(models_comp, "Bagging" = c(mean(abs(pred_bagging - test$price)),
                                                median(abs(pred_bagging - test$price)),
                                                min(abs(pred_bagging - test$price)),
                                                max(abs(pred_bagging - test$price)),
                                                mean(1-(abs(pred_bagging - test$price)/test$price))*100))


View(models_comp)

#Actual vs. Predicted
plot(test$price, pred_bagging, main = 'Bagging',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)
abline(a = 0, b = 1, col = "red", lwd = 2)

###Model 6 (SVM)-----
svm_model <- svm(T_price~.-price, data = train, kernel = "radial", cost = 1, gamma = 0.1)


# Perform grid search with cross-validation
set.seed(1122)
tune_out <- tune("svm", T_price~.-price, 
                 data = train, kernel = "polynomial",
                 ranges = list(cost = c(0.1, 1, 10),degree = c(2, 3, 4)))
summary(tune_out)

tune_out2 <- tune("svm", T_price~.-price, 
                  data = train, kernel = "polynomial",
                  ranges = list(cost = c(0.1, 1, 10)))

# Get the best parameter values
#cost 1/ degree 3
svm_model <- svm(T_price~.-price, data = train, kernel = "polynomial",cost = 1, degree = 3)

##Test the Model----------------------------------

predicted_prices <- predict(svm_model, newdata = test)
predicted_prices_SVM = (0.2*predicted_prices + 1)^5


#Absolute error mean, median, sd, max, min

models_comp <- rbind(models_comp, "SVM" = c(mean(abs(predicted_prices_SVM - test$price)),
                                            median(abs(predicted_prices_SVM - test$price)),
                                            min(abs(predicted_prices_SVM - test$price)),
                                            max(abs(predicted_prices_SVM - test$price)),
                                            mean(1-(abs(predicted_prices_SVM - test$price)/test$price))*100))


View(models_comp)


#Actual vs. Predicted
plot(test$price, predicted_prices_SVM, main = 'SVM',
     xlab = "Actual", ylab = "Prediction",cex.lab=1.5, cex.main=2.5)
abline(a = 0, b = 1, col = "red", lwd = 2)



