#ML Tutorial from Analytics Vidha
# https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/


setwd("//net.ucf.edu//ikm//Home//ja041718//Documents//R//loan data")

#load libraries and data
library(mlr)
library(corrplot)
train <- read.csv("train.csv", na.strings = c(""," ",NA))
test <- read.csv("test.csv", na.strings = c(""," ",NA))

#summarize data
summarizeColumns(train)

# ApplicantIncome and Coapplicant Income are highly skewed variables. How do we know that?
# Look at their min, max and median value. We'll have to normalize these variables.
hist(train$ApplicantIncome, breaks = 300, 
     main = "Applicant Income Chart",xlab = "ApplicantIncome")

hist(train$CoapplicantIncome, breaks = 100,
     main = "Coapplicant Income Chart",xlab = "CoapplicantIncome")

# LoanAmount, ApplicantIncome and CoapplicantIncome has outlier values, which should be 
# treated.

#notice all of the outliers above the concentration of data
boxplot(train$ApplicantIncome)

# Credit_History is an integer type variable. But, being binary in nature, 
# we should convert it to factor.
train$Credit_History = as.factor(train$Credit_History)
test$Credit_History = as.factor(test$Credit_History)
class(train$Credit_History)

# Notice the levels of dependents has a "3+" which will need re-labeling
summary(train)
levels(train$Dependents)[4] = "3"
levels(test$Dependents)[4] = "3"

                      #### Missing Value Imputation ####

#Impute function in MLR package allows you to impute to variable automatically based on
# its class, rather than calling its name individually.
imp <- impute(train, classes = list(factor = imputeMode(), integer = imputeMean()), 

#it also allows you to create a new dummy variable because sometimes the missing values
#contain some sort of trend. dummy.classes says for which classes should I create a 
# dummy variable. dummy.type says what should be the class of new dummy variables.
              dummy.classes = c("integer","factor"), dummy.type = "numeric")

imp1 = impute(test, classes = list(factor=imputeMode(), integer = imputeMean()),
              dummy.classes = c("integer", "factor"), dummy.type='numeric')

#We can now create new DFs with our imputed data
imp_train = imp$data
imp_test = imp1$data

#Check to see the distribution of the new dummy variables
summarizeColumns(imp_train)
summarizeColumns(imp_test)

#Since Married.dummy only exists in train set, it needs to be removed
imp_train$Married.dummy = NULL

                        ##### Feature Engineering #####

# Lets remove outliers from ApplicantIncome, CoapplicantIncome, LoanAmount. 
# There are many techniques to remove outliers. Here, we'll cap all the large values in
# these variables and set them to a threshold value as shown below

#for train data set
cd <- capLargeValues(imp_train, target = "Loan_Status",cols = c("ApplicantIncome"),
                     threshold = 40000)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("CoapplicantIncome"),
                     threshold = 21000)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("LoanAmount"),
                     threshold = 520)

#rename the train data as cd_train
cd_train <- cd

#add a dummy Loan_Status (dep variable) column in test data
imp_test$Loan_Status <- sample(0:1,size = 367,replace = T)

cde <- capLargeValues(imp_test, target = "Loan_Status",cols = c("ApplicantIncome"),
                      threshold = 33000)
cde <- capLargeValues(cde, target = "Loan_Status",cols = c("CoapplicantIncome"),
                      threshold = 16000)
cde <- capLargeValues(cde, target = "Loan_Status",cols = c("LoanAmount"),
                      threshold = 470)

#renaming test data
cd_test <- cde

# These loops say - 'for every column name which falls column number 14 to 19 of 
# cd_train / cd_test data frame, if the class of those variables is numeric, take out
# the unique values from those columns as levels and convert them into a factor 
# (categorical) variables.

#convert numeric to factor - train
for (f in names(cd_train[, c(14:19)])) {
  if( class(cd_train[, c(14:19)] [[f]]) == "numeric"){
      levels <- unique(cd_train[, c(14:19)][[f]])
      cd_train[, c(14:19)][[f]] <- as.factor(factor(cd_train[, c(14:19)][[f]], 
                                                    levels = levels))
  }
}

#convert numeric to factor - test
for (f in names(cd_test[, c(13:18)])) {
  if( class(cd_test[, c(13:18)] [[f]]) == "numeric"){
    levels <- unique(cd_test[, c(13:18)][[f]])
    cd_test[, c(13:18)][[f]] <- as.factor(factor(cd_test[, c(13:18)][[f]], levels = levels))
  }
}

#Create new features:

#Total_Income
cd_train$Total_Income <- cd_train$ApplicantIncome + cd_train$CoapplicantIncome
cd_test$Total_Income <- cd_test$ApplicantIncome + cd_test$CoapplicantIncome

#Income by loan
cd_train$Income_by_loan <- cd_train$Total_Income/cd_train$LoanAmount
cd_test$Income_by_loan <- cd_test$Total_Income/cd_test$LoanAmount

#change variable class
cd_train$Loan_Amount_Term <- as.numeric(cd_train$Loan_Amount_Term)
cd_test$Loan_Amount_Term <- as.numeric(cd_test$Loan_Amount_Term)

#Loan amount by term
cd_train$Loan_amount_by_term <- cd_train$LoanAmount/cd_train$Loan_Amount_Term
cd_test$Loan_amount_by_term <- cd_test$LoanAmount/cd_test$Loan_Amount_Term

#While creating new features(if they are numeric), we must check their correlation 
# with existing variables as there are high chances often. Let's see if our new 
# variables too happens to be correlated:

# Split the columns of the train DF based on if they are numeric or factors
az <- split(names(cd_train), sapply(cd_train, function(x){ class(x)}))

#creating a data frame of numeric variables
xs <- cd_train[az$numeric]

#Create correlation matrix of numeric variables
corrplot(cor(xs), method = 'number')

#Since the new var total income is highly correlated with applicant income, it is
#not providing enough new info for modeling. So let's remove it
cd_train$Total_Income = NULL
cd_test$Total_Income = NULL

                        #### Machine Learning ####

# Create tasks for classification 
trainTask <- makeClassifTask(data = cd_train,target = "Loan_Status")
testTask <- makeClassifTask(data = cd_test, target = "Loan_Status")

#Examine contents of cd_train data.
trainTask

#Notice it reads loan_status "N" as a positive class. That needs to be modified
trainTask <- makeClassifTask(data = cd_train,target = "Loan_Status", positive = "Y")

#Normalize the numeric variables
trainTask = normalizeFeatures(trainTask, method="standardize")
testTask = normalizeFeatures(testTask, method="standardize")

# Before we start applying algorithms, we should remove the variables which are not 
# required.
trainTask <- dropFeatures(task = trainTask,features = c("Loan_ID"))


                    #### Logistic Regression ####
#create CV learner
logistic.learner <- makeLearner("classif.logreg",predict.type = "response")

#use 3-fold CV
cv.logistic <- crossval(learner = logistic.learner,task = trainTask,iters = 3,
                        stratify = TRUE,measures = acc,show.info = F)

#So, I've used stratified sampling with 3 fold CV. I'd always recommend you to use 
#stratified sampling in classification problems since it maintains the proportion of 
#target class in n folds. We can check CV accuracy by:

#cross validation accuracy
cv.logistic$aggr

#to see accuracy on each fold
cv.logistic$measures.test

#Now train the model and check the prediction accuracy on the test data.

#train model
fmodel = train(logistic.learner, trainTask)
getLearnerModel(fmodel)

#predict on test set
fpmodel = predict(fmodel, testTask)
fpmodel

#create submission for Hackathon
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = fpmodel$data$response)
#79.16% accuracy, which at least means the model is stable since the 
# CV score on the training set and the Hackathon score are similar.


                          #### Decision Tree ####
library(rpart)
#make tree learner
makeatree = makeLearner("classif.rpart", predict.type="response")

#set 3-fold CV
set_cv = makeResampleDesc("CV", iters=3L)

#Search for hyperparameters
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)

# As you can see, I've set 3 parameters: 
# minsplit represents the minimum number of observation in a node for a split to take 
# place. 
# minbucket says the minimum number of observation I should keep in terminal nodes. 
# cp is the complexity parameter. The lesser it is, the tree will learn more 
# specific relationsin the data which might result in overfitting.

#do a grid search
gscontrol <- makeTuneControlGrid()

#hypertune the parameters
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = trainTask, 
                      par.set = gs, control = gscontrol, measures = acc)

#check best parameters
stune$x

#CV results
stune$y

# Using setHyperPars function, we can directly set the best parameters as 
# modeling parameters in the algorithm.

#using hyperparameters for modeling
t.tree <- setHyperPars(makeatree, par.vals = stune$x)

#train the model
t.rpart <- train(t.tree, trainTask)
getLearnerModel(t.rpart)

#make predictions
tpmodel <- predict(t.rpart, testTask)

#create a submission file
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = tpmodel$data$response)

#Like the log regression, it only got 79.1% accuracy.



                            #### Random Forest ####

#create a learner
rf <- makeLearner("classif.randomForest", predict.type = "response", 
                  par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
  importance = TRUE
)

#set tunable parameters
#grid search to find hyperparameters
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#let's do random search for 50 iterations
rancontrol <- makeTuneControlRandom(maxit = 50L)

# Though, random search is faster than grid search, but sometimes it turns out to be 
# less efficient. In grid search, the algorithm tunes over every possible combination
# of parameters provided. In a random search, we specify the number of iterations and 
#it randomly passes over the parameter combinations. In this process, it might miss out 
# some important combination of parameters which could have returned maximum accuracy, 
# who knows.

#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = trainTask, 
                        par.set = rf_param, control = rancontrol, measures = acc)

#cv accuracy
rf_tune$y

#best parameters
rf_tune$x

#Build RF model
#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rforest <- train(rf.tree, trainTask)
getLearnerModel(t.rpart)

#make predictions
rfmodel <- predict(rforest, testTask)

#submission file
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = rfmodel$data$response)

#Once again, only 79.14% accuracy.