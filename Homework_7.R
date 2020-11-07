# Library Functions
library(caret)
library(ggplot2)
library(tidyverse)

###################################### EDA #####################################
train <- read.csv("hm7-Train.csv")
test <- read.csv("hm7-test.csv")
glimpse(train)
glimpse(test)

nearZeroVar(train, saveMetrics = TRUE)
train.drop_nzv <- train[,-nearZeroVar(train)]
test.drop_nzv <- test[,-nearZeroVar(test)]
glimpse(train.drop_nzv)

############# Recast as factors ###################
train.drop_nzv$readmitted <- as.factor(train.drop_nzv$readmitted)
train.drop_nzv$admission_type <- as.factor(train.drop_nzv$admission_type)
train.drop_nzv$discharge_disposition <- as.factor(train.drop_nzv$discharge_disposition)
train.drop_nzv$admission_source <- as.factor(train.drop_nzv$admission_source)
train.drop_nzv$diagnosis <- as.factor(train.drop_nzv$diagnosis)
train.drop_nzv$A1Cresult <- as.factor(train.drop_nzv$A1Cresult)
train.drop_nzv$metformin <- as.factor(train.drop_nzv$metformin)
train.drop_nzv$glipizide <- as.factor(train.drop_nzv$glipizide)
train.drop_nzv$glyburide <- as.factor(train.drop_nzv$glyburide)
train.drop_nzv$pioglitazone <- as.factor(train.drop_nzv$pioglitazone)
train.drop_nzv$rosiglitazone <- as.factor(train.drop_nzv$rosiglitazone)
train.drop_nzv$insulin <- as.factor(train.drop_nzv$insulin)
train.drop_nzv$diabetesMed <- as.factor(train.drop_nzv$diabetesMed)


test.drop_nzv$readmitted <- as.factor(test.drop_nzv$readmitted)
test.drop_nzv$admission_type <- as.factor(test.drop_nzv$admission_type)
test.drop_nzv$discharge_disposition <- as.factor(test.drop_nzv$discharge_disposition)
test.drop_nzv$admission_source <- as.factor(test.drop_nzv$admission_source)
test.drop_nzv$diagnosis <- as.factor(test.drop_nzv$diagnosis)
test.drop_nzv$A1Cresult <- as.factor(test.drop_nzv$A1Cresult)
test.drop_nzv$metformin <- as.factor(test.drop_nzv$metformin)
test.drop_nzv$glipizide <- as.factor(test.drop_nzv$glipizide)
test.drop_nzv$glyburide <- as.factor(test.drop_nzv$glyburide)
test.drop_nzv$pioglitazone <- as.factor(test.drop_nzv$pioglitazone)
test.drop_nzv$rosiglitazone <- as.factor(test.drop_nzv$rosiglitazone)
test.drop_nzv$insulin <- as.factor(test.drop_nzv$insulin)
test.drop_nzv$diabetesMed <- as.factor(test.drop_nzv$diabetesMed)

# Displays the number of missing values by column.
map(train.drop_nzv, ~sum(is.na(.)))
high_missing_columns <- c("payer_code", "medical_specialty")
missing_indices <- which(high_missing_columns %in% colnames(train.drop_nzv))
missing_indices
train.transformed <- train.drop_nzv[,-missing_indices]
test.transformed <- test.drop_nzv[,-missing_indices]
glimpse(train.transformed)

################################# Predict ######################################
formula = readmitted ~ gender + age + time_in_hospital + num_lab_procedures + num_procedures


############ Random Forest ##############

train_control.rf <- trainControl(method = "boot",
                                 number = 5,)

train.fit.lm <- train(formula, 
                      data = train.transformed, 
                      method = 'rf',
                      tuneLength = 4)
train.fit.lm
prediction_matrix <- predict(train.fit.lm, test.transformed, type = "prob")
predictions <- prediction_matrix[, 1]
predictions
output <- cbind.data.frame(test$patientID, predictions)
names(output) = c("patientID","predReadmit")
output
write.csv(output, 'Submission1', row.names = FALSE)
