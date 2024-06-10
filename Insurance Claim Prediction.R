#Install Packages

install.packages("IRkernel")
IRkernel::installspec()
install.packages("formattable")
install.packages("PerformanceAnalytics")
install.packages("corrplot")
install.packages("stringr")
install.packages("visdat")
install.packages("ggplot2")
install.packages("RColorBrewer")
install.packages("dplyr")
install.packages("Hmisc")
install.packages("magrittr")
install.packages("ROSE")
install.packages("scatterplot3d")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("C50")
install.packages("randomForest")
install.packages("RANN")
install.packages("scatterplot3d")


#clear global environment objects and console area
rm(list=ls())
cat("\014")

#set RNG seed
set.seed(123)

#get working directory for reference
print(paste("WORKING DIRECTORY: ",getwd()))


#Load Packages

library("formattable")
library("PerformanceAnalytics")
library("corrplot")
library("stringr")
library("visdat")
library("ggplot2")
library("RColorBrewer")
library("dplyr")
library("Hmisc")
library("magrittr")
library("ROSE") #for balancing the dataset
library("lattice")
library("caret")
library("C50") #for the decision tree
library("randomForest")
library("RANN")
library("scatterplot3d")


#------------------------------------------------------------------------------------


#Dataset Source: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification

#Read train and test dataset csv from local files, upload csv files first
trainset <- read.csv("train.csv")

print("Dataset is saved in a dataframes called 'trainset'")


#Randomise the entire data set
trainset <- trainset[sample(1:nrow(trainset)),]


#-------------------------EDA---------------------

#This section checks for inconsistencies, nulls and repetitions in the dataset

summary(trainset)
print(paste("Number of duplicated rows in dataset=", sum(duplicated(trainset))))
print(paste("Number of complete rows in dataset=",sum(complete.cases(trainset))))
print(paste("Number of incomplete rows in dataset=",sum(!complete.cases(trainset))))



#-------------------------DATA PREPARATION---------------------------------

#There are 2 columns in the dataset with strings
head(trainset[,c("max_power", "max_torque")])


#This section splits the columns into 2 columns each
dataset_toRemoveStrings <- trainset
dataset_toRemoveStrings[c('max_torque_nm', 'max_torque_rpm')] <- c(str_split_fixed(trainset$max_torque, '@', 2))
dataset_toRemoveStrings[c('max_power_bhp', 'max_power_rpm')] <- c(str_split_fixed(trainset$max_power, '@', 2))


#This section removes the strings from the columns
dataset_remove_strings <- dataset_toRemoveStrings %>%
  mutate(max_torque_nm = str_remove(max_torque_nm, "Nm"),
         max_torque_rpm = str_remove(max_torque_rpm, "rpm"),
         max_power_bhp = str_remove(max_power_bhp, "bhp"),
         max_power_rpm = str_remove(max_power_rpm, "rpm"))


#This section converts the columns to numeric fields
str_to_numeric_dataset <- dataset_remove_strings %>% mutate_at(c('max_torque_nm',
                                                                 'max_torque_rpm',
                                                                 'max_power_bhp',
                                                                 'max_power_rpm'), as.numeric)


#This section drops the "policy_id" column and the "max_torque" and "max_power"
trainset_updated <- select(str_to_numeric_dataset, -(c("policy_id","max_torque","max_power")))

head(trainset_updated)


#Function to categorise data into symbolic and numeric
fieldCategories<-function(dataset){
  discrete_bins<-6
  field_types<-vector()
  field_type<-vector()
  for (i in 1:ncol(dataset)){
    if (is.numeric(dataset[,i])==TRUE){
      #191020NRT use R hist() function to create 10 bins
      histogramAnalysis<-hist(dataset[,i], breaks = 10, plot=FALSE)
      bins<-histogramAnalysis$counts/length(dataset[,i])*100  # Convert to %
      
      graphTitle<-"AUTO:"
      
      #If the number of bins with less than 1% of the values is greater than the cutoff
      #then the field is deterimed to be a discrete value
      
      if (length(which(bins<1.0))>discrete_bins){
        type <- "DISCRETE"
        field_type[i]<-"DISCRETE"
        field_types <- append(field_types,type)
      }
      
      else{
        type <- "CONTINUOUS"
        field_type[i]<-"CONTINUOUS"
        field_types <- append(field_types,type)
      }
      
      #Type of field is the chart name
      hist(dataset[,i], breaks = 10, plot=TRUE,
           main=paste(graphTitle,field_type[i]),
           xlab=names(dataset[i]),ylab="Number of Records",
           yaxs="i",xaxs="i",border = NA)
      #type <- "NUMERIC"
      #category <- cat(i,'',names(dataset[i]),':',type,'\n')
      #field_types <- append(field_types,type)
    }
    else{
      type <- "SYMBOLIC"
      #category <- cat(i,'',names(dataset[i]),':',type,'\n')
      field_types <- append(field_types,type)
    }
  }
  return(field_types)
}

field_types <- fieldCategories(trainset_updated)

print(field_types)



#CATEGORIZE DATASET INTO SYMBOLIC, DISCRETE AND CONTINUOUS

discrete <- names(trainset_updated)[field_types=="DISCRETE"]
continuous <- names(trainset_updated)[field_types=="CONTINUOUS"]
symbolic <- names(trainset_updated)[field_types=="SYMBOLIC"]

numeric <- append(discrete, continuous) #Merges the discrete and continuous data to numeric

print(paste("Number of discrete fields:",length(discrete)))
print(discrete)

cat("\n")
print(paste("Number of continuous fields:",length(continuous)))
print(continuous)

cat("\n")
print(paste("Number of symbolic fields:",length(symbolic)))
print(symbolic)



#--------------------ONE-HOT ENCODING---------------------------------

#Function for one-hot encoding 
oneHotEncoding<-function(dataset,field_types){
  
  catagorical<-data.frame()
  
  categorical_fields<-names(dataset)
  
  # for each field
  for (field in categorical_fields){
    
    # Convert into factors. A level for each unique string
    ffield<-factor(dataset[,field])
    
    # Check if too many unique values to encode
    if (nlevels(ffield) > 25) {
      stop(paste("Practical Business Analytics - too many literals in:",
                 field,
                 nlevels(ffield)))
    }
    
    # Check if just one value!
    if (nlevels(ffield) ==1) {
      stop(paste("Practical Business Analytics - field stuck at a single value:",
                 field))
    }
    
    # 1-hot encoding. A new column for each unique "level"
    xx<-data.frame(model.matrix(~ffield+0, data=ffield))
    
    names(xx)<-gsub("ffield",field,names(xx))
    
    # If 2 unique values, then can encode as a single "binary" column
    if (ncol(xx)==2){
      xx<-xx[,-2,drop=FALSE]
      names(xx)<-field  # Field name without the value appended
    }
    
    catagorical<-as.data.frame(append(catagorical,xx))
    
  } #endof for()
  return (catagorical)
  
}

nums <- unlist(lapply(trainset_updated, is.numeric), use.names = FALSE)


cattrain <- trainset_updated[, !(names(trainset_updated) %in% names(trainset_updated[, nums]))]
head(cattrain)

trainset_for_encoding <- trainset_updated[, !(names(trainset_updated) %in% names(cattrain))]
head(trainset_for_encoding)

cattrain <- oneHotEncoding(cattrain[,!names(cattrain) %in% c("policy_id")])
head(cattrain)

for (column in names(cattrain)){
  trainset_for_encoding[column] <- cattrain[column]
}
head(trainset_for_encoding)

pairs(trainset_for_encoding[ , nums])


for (x in names(trainset_for_encoding)) {
  counts <- table(trainset_for_encoding[[x]])
  barplot(counts, main=c(x," Distribution"),
          xlab=x)
}

print(table(trainset_for_encoding["is_claim"]))




#-------------------------------CORRELATION AND REDUNDANCY---------------------

#Function to check for reduncancy in the dataset
redundantFields<-function(dataset,cutoff){
  
  print(paste("Before redundancy check Fields=",ncol(dataset)))
  
  #Remove any fields that have a stdev of zero (i.e. they are all the same)
  xx<-which(apply(dataset, 2, function(x) sd(x, na.rm=TRUE))==0)+1
  
  if (length(xx)>0L)
    dataset<-dataset[,-xx]
  
  #Kendall is more robust for data do not necessarily come from a bivariate normal distribution.
  cr<-cor(dataset, use="everything")
  #cr[(which(cr<0))]<-0 #Positive correlation coefficients only
  corrPlot(cr)
  
  correlated<-which(abs(cr)>=cutoff,arr.ind = TRUE)
  list_fields_correlated<-correlated[which(correlated[,1]!=correlated[,2]),]
  
  if (nrow(list_fields_correlated)>0){
    
    print("Following fields are correlated")
    print(list_fields_correlated)
    
    # 240220nrt print list of correlated fields as namesß
    for (i in 1:nrow(list_fields_correlated)){
      print(paste(names(dataset)[list_fields_correlated[i,1]],"~", names(dataset)[list_fields_correlated[i,2]]))
    }
    
    #We have to check if one of these fields is correlated with another as cant remove both!
    v<-vector()
    numc<-nrow(list_fields_correlated)
    for (i in 1:numc){
      if (length(which(list_fields_correlated[i,1]==list_fields_correlated[i:numc,2]))==0) {
        v<-append(v,list_fields_correlated[i,1])
      }
    }
    print("Removing the following fields")
    print(names(dataset)[v])
    
    return(dataset[,-v]) #Remove the first field that is correlated with another
  }
  return(dataset)
}

corrPlot<-function(cr){
  
  #Defines the colour range
  col<-colorRampPalette(c("green", "red"))
  
  #To fit on screen, convert field names to a numeric
  rownames(cr)<-1:length(rownames(cr))
  colnames(cr)<-rownames(cr)
  
  corrplot::corrplot(abs(cr),method="square",
                     order="FPC",
                     cl.ratio=0.2,
                     cl.align="r",
                     tl.cex = 0.6,cl.cex = 0.6,
                     cl.lim = c(0, 1),
                     mar=c(1,1,1,1),bty="n")
}

numtrain <- trainset_for_encoding[, nums]

redundantFields(numtrain, 0.8)



#Dropping some columns
drop <- c("airbags", "gear_box", "length")
trainset_after_preprocessing <- trainset_for_encoding[, !(names(trainset_for_encoding) %in% drop)]



#split training dataset into training and validation
training_records <- round(nrow(trainset_after_preprocessing)*(70/100))
trainset_for_split <- sample(trainset_after_preprocessing)

train<-trainset_for_split[1:training_records,] #Training data
valid<-trainset_for_split[-(1:training_records),] #Testing/validation data

head(train)
ncol(train)


#----------------------------REBALANCING---------------------------
#https://CRAN.R-project.org/package=ROSE

#Plot to show the unbalanced nature of the dataset
barplot(table(train$is_claim), main="is_claim Imbalance",
        xlab="is_claim", ylab="Frequency")



#Display the proportion of the "is_claim" column
print("The proportion of unclaimed insurance to claimed insurance is: ")
prop.table(table(train$is_claim))
table(train$is_claim)


#These variable assignments are used for the re-balancing of the dataset
is_claim<-matrix(table(train$is_claim))
not_claimed <- is_claim[1]
claimed<- is_claim[2]
print(table(trainset$is_claim))


#-----------------OVERSAMPLING--------------------

#Using number of unclaimed
oversamp_frac <- 0.53  #Oversampling fraction. This represents the desired proportion of the minority class
new_oversamp <- not_claimed / oversamp_frac #This calculates the desired number of instances of the minority class

oversamp_result <- ovun.sample(formula= is_claim ~ ., data=train,
                               method="over", N=new_oversamp,
                               seed=2018)

oversampled_dataset <- oversamp_result$data
prop.table(table(oversampled_dataset$is_claim))

#Confirm size of dataset after oversampling
table(oversampled_dataset$is_claim)

#Visualise oversampled dataset
barplot(table(oversampled_dataset$is_claim), main="is_claim oversampling",
        xlab="is_claim")


#------------------UNDERSAMPLING----------------------
#Using number of claimed
undersamp_frac <- 0.46  #Undersampling fraction. This represents the desired proportion of the majority class
new_undersamp <- claimed / undersamp_frac #This calculates the desired number of instances of the majority class

undersamp_result <- ovun.sample(formula= is_claim ~ ., data=train,
                                method="under", N=new_undersamp,
                                seed=2018)
undersampled_dataset <- undersamp_result$data
prop.table(table(undersampled_dataset$is_claim))


#Confirm size of dataset after undersampling
table(undersampled_dataset$is_claim)

#Visualise undersampled dataset
barplot(table(undersampled_dataset$is_claim), main="is_claim undersampling",
        xlab="is_claim")


#-----------------------------EVALUATIONS-------------------------------------------


#All evaluation metrics are calculated in this function

modelEvaluations <- function(dataModel, testModel, modelName){
  conf_matrix <- confusionMatrix(dataModel, testModel$is_claim, positive="0")
  
  accuracy <- conf_matrix$overall['Accuracy']
  
  precision <- conf_matrix$byClass['Pos Pred Value']
  
  recall <- conf_matrix$byClass['Sensitivity']
  
  f1Score <- 2 * (precision * recall) / (precision + recall)
  
  evaluations <- cat("The following are the evaluation scores for the", modelName ,"model: \nAccuracy: ", accuracy, 
                   "\nPrecision: ", precision, "\nRecall: ", recall, "\nF1 Score: ", f1Score)
  
  return(evaluations)
  
}



#--------------------------------------------------------CHIAMAKA CHUDE------------------------------------------

#The decision tree and random forest models were put in functions as they would be run multiple times on
#different datasets, and parameters would be fine-tuned to inprove performance. They were put in functions
#for easier understanding, better code readability, and conservation of space.


#-------------------------------DECISION TREE---------------------------------------
#Source: https://cran.r-project.org/package=C50

#The "is_claim" column would need to be converted to a factor in order to run the models and perform model evaluations
valid$is_claim <- as.factor(valid$is_claim)
oversampled_dataset$is_claim <- as.factor(oversampled_dataset$is_claim)
undersampled_dataset$is_claim <- as.factor(undersampled_dataset$is_claim)



#Decision Tree model. This function takes in the training data and testing data as arguments
DecisionTreeModel <- function(trainingset, testingset){
  
  model <- C5.0(is_claim ~ ., data = trainingset, trials=25) #model
  predictions <- predict(model, newdata = testingset) #predictions on testing data
  return(predictions)
}



#Decision Tree modelling and evaluation with the oversampled dataset. 

#Model predictions are saved in this variable
oversampledDTModel <- DecisionTreeModel(oversampled_dataset, valid)

#The model predictions are evaluated against the actual values in the testing data
oversampledDT_evaluations <- modelEvaluations(oversampledDTModel, valid, "Oversampled Decision Tree")
print(oversampledDT_evaluations)



#Decision Tree modelling and evaluation with undersampled dataset. 

#Model predictions are saved in this variable
undersampledDTModel <- DecisionTreeModel(undersampled_dataset, valid)

#The model predictions are evaluated against the actual values in the testing data
undersampledDT_evaluations <- modelEvaluations(undersampledDTModel, valid, "Undersampled Decision Tree")
print(undersampledDT_evaluations)



#-------------------------------RANDOM FOREST--------------------------------
#Source: https://cran.r-project.org/package=randomForest

#Random Forest model. This function takes in the training data and testing data as arguments
randomForestModel <- function(trainingset, testingset){
  
  model <- randomForest(is_claim ~ ., data = trainingset, mtry = 3, nodesize = 5) #model
  predictions <- predict(model, testingset) #predictions on testing data
  return(predictions)
}


#Random Forest modelling and evaluation with oversampled dataset. 

#Model predictions are saved in this variable
oversampledRFModel <- randomForestModel(oversampled_dataset, valid)

#The model predictions are evaluated against the actual values in the testing data
oversampledRF_evaluations <- modelEvaluations(oversampledRFModel, valid, "Oversampled Random Forest")
print(oversampledRF_evaluations)



#Random Forest modelling and evaluation with undersampled dataset. 

#Model predictions are saved in this variable
undersampledRFModel <- randomForestModel(undersampled_dataset, valid)

#The model predictions are evaluated against the actual values in the testing data
undersampledRF_evaluations <- modelEvaluations(undersampledRFModel, valid, "Undersampled Random Forest")
print(undersampledRF_evaluations)

#-----------------------------------------------ADDITIONAL PLOTS---------------------------------------------------

#Accuracy scores for the decision tree model - CHIAMAKA
oversampled_accuracy_scores <- c(88.5, 90.7, 91.2, 91.4, 91.7)
undersampled_accuracy_scores <- c(55.4, 63.6, 63.9, 63.9, 63.9)
trials <- c(5, 10, 15, 20, 25)


# Plot
plot(trials, oversampled_accuracy_scores, type = "o", col = "purple", pch = 16,
     xlab = "Trials", ylab = "Accuracy",
     main = "Decision Tree Accuracy Comparison", 
     ylim=c(50,95))

lines(trials, undersampled_accuracy_scores, type = "o", col = "orange", pch = 16, lty = 2)
legend("bottomright", legend = c("Oversampled", "Undersampled"),
       col = c("purple", "orange"), pch = 16, lty = c(1, 2))
grid()
