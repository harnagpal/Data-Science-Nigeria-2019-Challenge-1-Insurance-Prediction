# ready the csv file

setwd("D:/Harish/Zindi/Nigeria_Insurance_Prediction")
getwd()

# load packages
library(dplyr)
library(readr)
library(stringr)
library(caret)
#library(data.table)
library(mltools)
library(plyr) # for rbind
#library(randomForest)
library(lubridate) # for dates
library(tidyr)
#library(tibble)
#library(purrr)
library(Matrix)
library(sqldf) # for running sql queries
library(catboost)


train1 = read.csv("train_data.csv")
test1 = read.csv("test_data.csv")

head(train1, n = 5)

summary(train1)
summary(test1)

train1$flag <- 'traindata'
test1$flag <- 'testdata'


# Number of windows around with . value around 50%
# Building.Dimension, NA 106,  - take mean
#Garden - NA 7 - Replace with V
#Geo code - 102 Blank -- Replace with mode
#Date of Occupancy - 508 blank  - Replace with round(mode)

summary(test1)

nrow(train1) #7160
nrow(test1) # 3069

#combine both data sets
combined  <- rbind.fill(train1, test1) 
summary(combined)
nrow(combined) #10229

str(combined)

#Feaure engineering


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

combined$Building_Type <- as.factor(combined$Building_Type)
combined$Residential <- as.factor(combined$Residential)

DoA <- getmode(combined$Date_of_Occupancy)
combined$Date_of_Occupancy[is.na(combined$Date_of_Occupancy)] <- DoA

combined$dur <- combined$YearOfObservation -  combined$Date_of_Occupancy
combined$dur = (combined$dur - mean(combined$dur)) / sd(combined$dur)

geocode <- getmode(combined$Geo_code)
combined$Geo_code[is.na(combined$Geo_code)] <- geocode

combined$Garden <- as.character(combined$Garden)
combined$Garden[which(combined$Garden == "")] <- "V"
combined$Garden <- as.factor(combined$Garden)

head(combined)
summary(combined)
str(combined)

head(combined)
summary(combined)

#str(combined$Building.Dimension)
#check NAs
na_X <- sapply(combined, function(y) sum(is.na(y)))
na_X

#Rounding Insured_period
rnd <- combined$Insured_Period
rnd <- round(rnd, digits=1)

unique(rnd)

rnd <- replace(rnd, rnd == 0.0, 1)
rnd <- replace(rnd, rnd == 0.1, 2)
rnd <- replace(rnd, rnd == 0.2, 3)
rnd <- replace(rnd, rnd == 0.3, 4)
rnd <- replace(rnd, rnd == 0.4, 5)
rnd <- replace(rnd, rnd == 0.5, 6)
rnd <- replace(rnd, rnd == 0.6, 7)
rnd <- replace(rnd, rnd == 0.7, 8)
rnd <- replace(rnd, rnd == 0.8, 9)
rnd <- replace(rnd, rnd == 0.9, 10)
rnd <- replace(rnd, rnd == 1.0, 12)
combined$Frequency <- rnd

str(combined)

combined$Garden <- as.character(combined$Garden)
combined$Garden[which(combined$Garden == "V")] <- 1
combined$Garden[which(combined$Garden == "O")] <- 0
combined$Garden <- as.integer(combined$Garden)

combined$Building_Painted <- as.character(combined$Building_Painted)
combined$Building_Painted[which(combined$Building_Painted == "N")] <- 1
combined$Building_Painted[which(combined$Building_Painted == "V")] <- 0
combined$Building_Painted <- as.integer(combined$Building_Painted)

combined$Building_Fenced <- as.character(combined$Building_Fenced)
combined$Building_Fenced[which(combined$Building_Fenced == "N")] <- 1
combined$Building_Fenced[which(combined$Building_Fenced == "V")] <- 0
combined$Building_Fenced <- as.integer(combined$Building_Fenced)

combined$Settlement <- as.character(combined$Settlement)
combined$Settlement[which(combined$Settlement == "U")] <- 1
combined$Settlement[which(combined$Settlement == "R")] <- 0
combined$Settlement <- as.integer(combined$Settlement)

combined$Residential <- as.character(combined$Residential)
combined$Residential <- as.integer(combined$Residential)

combined$Building_Type <- as.character(combined$Building_Type)
combined$Building_Type <- as.integer(combined$Building_Type)

combined$YearOfObservation <- NULL
combined$Date_of_Occupancy <- NULL
# combined$NumberOfWindows <- NULL
combined$Insured_Period <- NULL

#combined$ID <- NULL
combined$Frequency <- as.factor(as.character(combined$Frequency))

head(combined, n=20)

str(combined)



head(combined, head =10)
#separating train and test data after feature engineering
Otrain <- subset(combined, flag=='traindata')
Otest <- subset(combined, flag=='testdata')
Otrain$flag <- NULL
Otest$flag <- NULL

mnBD <- median(Otrain$Building.Dimension, na.rm = TRUE)
Otrain$Building.Dimension[is.na(Otrain$Building.Dimension)] <- mnBD

Otrain$Building.Dimension = (Otrain$Building.Dimension - mean(Otrain$Building.Dimension)) / sd(Otrain$Building.Dimension)


mnBD_tst <- median(Otest$Building.Dimension, na.rm = TRUE)
Otest$Building.Dimension[is.na(Otest$Building.Dimension)] <- mnBD_tst

Otest$Building.Dimension = (Otest$Building.Dimension - mean(Otest$Building.Dimension)) / sd(Otest$Building.Dimension)


names(Otrain)
nrow(Otrain)

str(Otrain)
str(Otest)

#Otest
###############
# Otrain <- qTrain
# Otest <- qTest

#train1$Response <- as.numeric(train1$Response)
Otrain$Customer.Id <- NULL

TstCusId <- Otest$Customer.Id # we will use this value at the time of submission
Otest$Customer.Id <- NULL
######Catboost
# using only important fields after I ran varaible importance
Otrain <- select(Otrain, Claim, Residential, Building.Dimension, Building_Type, dur, Frequency, Geo_Code,Settlement,NumberOfWindows)
Otest <- select(Otest, Claim, Residential, Building.Dimension, Building_Type, dur, Frequency, Geo_Code,Settlement,NumberOfWindows)

set.seed(7)



y_train <- unlist(Otrain[c('Claim')])
X_train <- Otrain %>% select(-Claim)
y_valid <- unlist(Otest[c('Claim')])
X_valid <- Otest %>% select(-Claim)



train_pool <- catboost.load_pool(data = X_train, label = y_train)
test_pool <- catboost.load_pool(data = X_valid, label = y_valid)

params <- list(iterations=500,
               learning_rate=0.01,
               depth=10,
               loss_function='RMSE',
               eval_metric='AUC',
               random_seed = 55,
               od_type='Iter',
               metric_period = 50,
               od_wait=20,
               bootstrap_type = 'Bernoulli',
               use_best_model=TRUE)

model <- catboost.train(learn_pool = train_pool, params = params)

y_pred=catboost.predict(model,test_pool)


#Use below code for Submission

#TstCusId <- as.data.frame(TstCusId)
#test_predictions_3 <- cbind(TstCusId, y_pred) 

#%>%   rename(Survived = rf_pred_3)

#write.csv(test_predictions_3, file = 'final_pred_Claim_CTBST_0110_BER_2.csv', row.names = T)



# xgbpred_r <- ifelse (y_pred > 0.50,1,0)
# xgbpred_r <- as.factor(xgbpred_r)
# test.label <- as.factor(y_valid)
# #confusion matrix
# confusionMatrix (xgbpred_r, test.label)

#TstCusId <- as.data.frame(TstCusId)
#test_predictions_3 <- cbind(TstCusId, y_pred) 

#%>%   rename(Survived = rf_pred_3)

#write.csv(test_predictions_3, file = 'final_pred_Claim_CTBST_0110_BER_2.csv', row.names = T)

#enssambling
#ber1 = read.csv("final_pred_Claim_CTBST_0110_BER.csv")
#ber2 = read.csv("final_pred_Claim_CTBST_0110_BER_2.csv")
#names(ber1)
#names(ber2)

#ensm <- sqldf("select ber1.TstCusId, ber1.y_pred, ber2.y_pred_2 from ber1, ber2 where ber1.TstCusId = ber2.TstCusId" )

#ensm$y_pred_3 <- ensm$y_pred*.70 + ensm$y_pred_2*.30

#ensm$y_pred <- NULL
#ensm$y_pred_2 <- NULL

#write.csv(ensm, file = 'final_pred_Claim_CTBST_0110_ENSM_3.csv', row.names = T)


#summary(ensm)

#feature_importance
#catboost.get_feature_importance(model, 
#                               pool = NULL, 
#                               type = 'FeatureImportance',
#                               thread_count = -1)



#####end of catboost
























