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
##### Train ####
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
train.drop_nzv$age <- ordered(train.drop_nzv$age)

##### Train ####
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
test.drop_nzv$age <- ordered(test.drop_nzv$age)

# Displays the number of missing values by column.
train.drop_nzv %>% 
  map(~sum(is.na(.)))

train.transformed %>% 
  map(~sum(is.na(.)))

high_missing_columns <- c("payer_code", "medical_specialty")

# Drop Columns with too many missing variables.
train.drop_nzv %>% 
  select (-high_missing_columns) -> train.transformed
test.drop_nzv %>% 
  select (-high_missing_columns) -> test.transformed
glimpse(train.transformed)

# Handle NA
train.transformed %>% 
  mutate_at(c("race"), replace_na, "Caucasian") ->
  train.transformed

test.transformed %>% 
  mutate_at(c("race"), replace_na, "Caucasian") ->
  test.transformed

# Fact lump
test.transformed$readmitted = -1
both <- rbind(train.transformed, test.transformed)

both %>% 
  mutate_at("diagnosis", fct_lump_min, 1000) %>% 
  mutate_at("diagnosis", replace_na, "Other") ->
  both.transformed

train.lumped = both.transformed %>% filter(!is.na(readmitted))
test.lumped = both.transformed %>% filter(is.na(readmitted))

ggplot(train.lumped) +
  geom_bar(mapping = aes(diagnosis, fill=readmitted), 
                 stat = 'count',
                 col='Black')

################################# Predict ######################################
formula = readmitted ~ gender + 
  age + 
  time_in_hospital + 
  num_lab_procedures + 
  num_procedures + 
  admission_type + 
  number_emergency +
  diagnosis +
  race


train_control <- trainControl(method = "boot",
                                 number = 25,
                                 verboseIter = TRUE)


############ Random Forest ##############
tune_grid.rf <- expand.grid(mtry=2)

train.fit.rf <- train(formula, 
                      data = train.lumped, 
                      method = 'rf',
                      tuneLength = 3,
                      trControl = train_control,
                      verbose = TRUE,
                      tuneGrid = tune_grid.rf)
train.fit.rf
prediction_matrix <- predict(train.fit.rf, test.lumped, type = "prob")

############ Neural Network #############
tune_grid.nn <- expand.grid(decay=c(0.5, 0.1, 0.125, 0.15),
                            size=c(2,3,4))

train.fit.nn <- train(formula,
                      data = train.lumped,
                      method = 'nnet',
                      tuneLength = 4,
                      verbose = TRUE,
                      trControl = train_control,
                      tuneGrid = tune_grid.nn)
train.fit.nn
# Best Accuracy: 0.5838946

prediction_matrix <- predict(train.fit.nn, test.lumped, type = "prob")

############ Output #####################
predictions <- prediction_matrix[, 1]
predictions
output <- cbind.data.frame(test$patientID, predictions)
names(output) = c("patientID","predReadmit")
output
write.csv(output, 'Submission2.csv', row.names = FALSE)
