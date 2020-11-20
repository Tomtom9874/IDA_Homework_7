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

train.reclassed <- train.drop_nzv
train.reclassed$readmitted  <- as.factor(ifelse(train.drop_nzv$readmitted == 0, "Not_Readmitted", "Readmitted"))
# Displays the number of missing values by column.
train.reclassed %>% 
  map(~sum(is.na(.)))



high_missing_columns <- c("payer_code", "medical_specialty")

# Drop Columns with too many missing variables.
train.reclassed %>% 
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
  both

both %>% 
  mutate_at("gender", fct_lump_n, 1) %>% 
  mutate_at("gender", replace_na, "Other") ->
  both.transformed

train.lumped = both.transformed %>% filter(!is.na(readmitted))
test.lumped = both.transformed %>% filter(is.na(readmitted))


ggplot(train.lumped) +
  geom_bar(mapping = aes(age, fill='blue'), 
                 stat = 'count',
                 col='Black')

glimpse(train.lumped)

################################# Predict ######################################
#formula = readmitted ~ gender + 
#  age + 
#  time_in_hospital + 
#  num_lab_procedures + 
#  num_procedures + 
#  admission_type + 
#  number_emergency +
#  diagnosis +
#  race

formula = readmitted ~ gender + 
  age + 
  time_in_hospital + 
  num_lab_procedures + 
  num_procedures + 
  race +
  diagnosis

formula <- readmitted ~ .

########### Train Control ###############
crossValidation <- trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 2,
                                verboseIter = TRUE,
                                summaryFunction=mnLogLoss,
                                classProbs = TRUE)

############ XGBoost ####################

train.fit.xgb <- train(formula, 
                      data = train.lumped, 
                      method = 'xgbTree',
                      trControl = crossValidation,
                      verbose = TRUE,
                      tuneLength = 3,
                      metric = 'logLoss',
                      nthreads = 1)
train.fit.xgb

# Best LogLoss: 0.6682219
prediction_matrix <- predict(train.fit.xgb, test.lumped, type = "prob")

############ Random Forest ##############
tune_grid.rf <- expand.grid(mtry=c(2,3))

train.fit.rf <- train(formula, 
                      data = train.lumped, 
                      method = 'rf',
                      trControl = crossValidation,
                      verbose = TRUE,
                      tuneGrid = tune_grid.rf,
                      metric = 'logLoss'
                      )
train.fit.rf

# Best LogLoss: 0.869935
prediction_matrix <- predict(train.fit.rf, test.lumped, type = "prob")





############### GLM ################
train.fit.glm <- train(formula, 
                      data = train.lumped, 
                      method = 'glm',
                      trControl = crossValidation,
                      family = 'binomial',
                      metric = 'logLoss'
)
train.fit.glm
prediction_matrix <- predict(train.fit.glm, test.lumped, type = "prob")
############ Neural Network #############
tune_grid.nn <- expand.grid(decay=c(.175),
                            size=c(3, 4))

train.fit.nn <- train(formula,
                      data = train.lumped,
                      method = 'nnet',
                      verbose = TRUE,
                      trControl = crossValidation,
                      tuneControl = tune_grid.nn,
                      metric = 'logLoss')
train.fit.nn
train.fit.nn$finalModel
# Best LogLoss: 0.6684635
# Best Submission LL: 0.77121
prediction_matrix <- predict(train.fit.nn, test.lumped, type = "prob")

############ SVM ########################

train.fit.svm <- train(formula,
                      data = train.lumped,
                      method = 'svm',
                      verbose = TRUE,
                      trControl = crossValidation,
                      metric = 'logLoss')
train.fit.svm

# Best LogLoss: 0.6684635
# Best Submission LL: 0.77121
prediction_matrix <- predict(train.fit.svm, test.lumped, type = "prob")

############ Output #####################
predictions <- prediction_matrix[, 1]
output <- cbind.data.frame(test$patientID, predictions)
names(output) = c("patientID","predReadmit")
write.csv(output, 'Submission11.csv', row.names = FALSE)
