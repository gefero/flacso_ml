## ---- message=FALSE------------------------------------------------------
library(caret)
library(tidyverse)
library(rpart)


## ------------------------------------------------------------------------
load('../data/EPH_2015_II.RData')

data$pp03i<-factor(data$pp03i, labels=c('1-SI', '2-No', '9-NS'))



data$intensi<-factor(data$intensi, labels=c('1-Sub_dem', '2-SO_no_dem', 
                                            '3-Ocup.pleno', '4-Sobreoc',
                                            '5-No trabajo', '9-NS'))

data$pp07a<-factor(data$pp07a, labels=c('0-NC',
                                        '1-Menos de un mes',
                                        '2-1 a 3 meses',
                                        '3-3 a 6 meses',
                                        '4-6 a 12 meses',
                                        '5-12 a 60 meses',
                                        '6-Más de 60 meses',
                                        '9-NS'))



data <- data %>%
        mutate(imp_inglab1=factor(imp_inglab1, labels=c('non_miss','miss')))


## ------------------------------------------------------------------------

df_train <- data %>%
        select(-p21)



## ------------------------------------------------------------------------
set.seed(1234)
tr_index <- createDataPartition(y=df_train$imp_inglab1,
                                p=0.8,
                                list=FALSE)


## ------------------------------------------------------------------------
train <- df_train[tr_index,]
test <- df_train[-tr_index,]


## ------------------------------------------------------------------------
fitControl <- trainControl(method = "none", classProbs = FALSE)


## ------------------------------------------------------------------------
cart_tune <- train(imp_inglab1 ~ . , 
                 data = df_train, 
                 method = "rpart2", 
                 trControl = fitControl,
                 tuneGrid = data.frame(maxdepth=3),
                 control = rpart.control(minsplit = 1,
                                         minbucket = 1,
                                        cp=0.00000001)
)


## ------------------------------------------------------------------------
plot(cart_tune$finalModel)
text(cart_tune$finalModel, pretty=1)


## ------------------------------------------------------------------------
library(rpart.plot)


## ------------------------------------------------------------------------
rpart.plot(cart_tune$finalModel)


## ------------------------------------------------------------------------
table(predict(cart_tune, df_train) , df_train$imp_inglab1)


## ------------------------------------------------------------------------
cart_tune <- train(imp_inglab1 ~ . , 
                 data = df_train, 
                 method = "rpart2", 
                 trControl = fitControl,
                 tuneGrid = data.frame(maxdepth=10),
                 control = rpart.control(cp=0.0001)
)


## ------------------------------------------------------------------------
rpart.plot(cart_tune$finalModel)


## ------------------------------------------------------------------------
table(predict(cart_tune, df_train) , df_train$imp_inglab1)


## ------------------------------------------------------------------------
set.seed(789)


## ------------------------------------------------------------------------
cv_index <- createFolds(y = train$imp_inglab1,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)


## ------------------------------------------------------------------------
fitControl <- trainControl(
        index=cv_index,
        method="cv",
        number=5
        )


## ------------------------------------------------------------------------
grid <- expand.grid(maxdepth=c(1, 2, 4, 8, 10, 15, 20))


## ----warning=FALSE-------------------------------------------------------
cart_tune <- train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "rpart2", 
                 trControl = fitControl,
                 tuneGrid = grid,
                 control = rpart.control(cp=0.000001)
)

cart_tune


## ------------------------------------------------------------------------
cart_final <- train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "rpart2", 
                 tuneGrid = data.frame(maxdepth=6),
                 control = rpart.control(cp=0.000001)
)


## ----fig.height=12, fig.width=20-----------------------------------------
rpart.plot(cart_final$finalModel)


## ------------------------------------------------------------------------
y_preds <- predict(cart_final, test)


## ------------------------------------------------------------------------
confusionMatrix(y_preds, test$imp_inglab1)


## ------------------------------------------------------------------------
set.seed(5699)
cv_index_rf <- createFolds(y=train$imp_inglab1,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

fitControlrf <- trainControl(
        index=cv_index_rf,
        method="cv",
        number=5,
        summaryFunction = twoClassSummary,
        classProbs=TRUE
        )

# Generar grid

grid_rf <- expand.grid(mtry=c(5,10,15, 25),
                       min.node.size=c(5,10,15,20),
                       splitrule='gini'
        )



## ----eval=FALSE, include=TRUE--------------------------------------------
## # Tunear
## t0 <- proc.time()
## rf_fit_orig <-  train(imp_inglab1 ~ . ,
##                  data = train,
##                  method = "ranger",
##                  trControl = fitControlrf,
##                  tuneGrid = grid_rf
## )
## proc.time() -  t0
## 
## #saveRDS(rf_fit_orig, '../models/rf_fit_orig.RDS')


## ----include=FALSE-------------------------------------------------------
rf_fit_orig <- readRDS('../models/rf_fit_orig.RDS')


## ------------------------------------------------------------------------
rf_fit_orig

y_preds_orig <- predict(rf_fit_orig, test)


## ------------------------------------------------------------------------
ggplot(train) + 
        geom_bar(aes(x=imp_inglab1))


## ------------------------------------------------------------------------
prop.table(table(y_preds_orig, test$imp_inglab1))


## ------------------------------------------------------------------------
model_weights <- ifelse(train$imp_inglab1 == "non_miss",
                        (1/table(train$imp_inglab1)[1]) * 0.5,
                        (1/table(train$imp_inglab1)[2]) * 0.5)

fitControlrf$seeds <- rf_fit_orig$control$seeds


## ------------------------------------------------------------------------
t0 <- proc.time()
rf_fit_wei <-  train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "ranger", 
                 trControl = fitControlrf,
                 tuneGrid = grid_rf,
                 weights = model_weights,
                 metric='ROC'
)
proc.time() -  t0

#saveRDS(rf_fit_wei, '../models/rf_fit_wei.RDS')



## ----include=FALSE-------------------------------------------------------
rf_fit_wei <- readRDS('../models/rf_fit_wei.RDS')


## ------------------------------------------------------------------------

# Nuevo fitControl con upsample

fitControlrf_imb <- trainControl(
        index=cv_index_rf,
        method="cv",
        number=5,
        summaryFunction = twoClassSummary,
        classProbs=TRUE,
        sampling = 'up'
        )



## ----eval=FALSE, include=TRUE--------------------------------------------
## 
## # Tunear upsample
## 
## fitControlrf_imb$seeds <- rf_fit_orig$control$seeds
## 
## t0 <- proc.time()
## rf_fit_up <-  train(imp_inglab1 ~ . ,
##                  data = train,
##                  method = "ranger",
##                  trControl = fitControlrf_imb,
##                  tuneGrid = grid_rf,
##                  metric='ROC'
## )
## proc.time() -  t0
## 
## saveRDS(rf_fit_up, '../models/rf_fit_up.RDS')


## ----include=FALSE-------------------------------------------------------
rf_fit_up <- readRDS('../models/rf_fit_up')



## ----eval=FALSE, include=TRUE--------------------------------------------
## # Tunear downsample
## 
## fitControlrf_imb$sampling <- "down"
## fitControlrf_imb$seeds <- rf_fit_orig$control$seeds
## 
## 
## t0 <- proc.time()
## rf_fit_down <-  train(imp_inglab1 ~ . ,
##                  data = train,
##                  method = "ranger",
##                  trControl = fitControlrf_imb,
##                  tuneGrid = grid_rf,
##                  metric='ROC'
## )
## proc.time() -  t0
## 
## #saveRDS(rf_fit_down, '../models/rf_fit_down.RDS')
## 


## ----include=FALSE-------------------------------------------------------
rf_fit_down <- readRDS('../models/rf_fit_down.RDS')


## ----eval=FALSE, include=FALSE-------------------------------------------
## rf_fit_smote <- readRDS('../models/rf_fit_smote.RDS')


## ------------------------------------------------------------------------
model_list <- list(original = rf_fit_orig,
                   weighted = rf_fit_wei,
                   down = rf_fit_down,
                   up = rf_fit_up)



## ------------------------------------------------------------------------
extract_conf_metrics <- function(model, data, obs){
        preds <- predict(model, data)        
        c<-confusionMatrix(preds, obs)
        results <- c(c$overall[1], c$byClass)
        return(results)
}



## ------------------------------------------------------------------------

model_metrics <- model_list %>%
        map(extract_conf_metrics, data=test, obs = test$imp_inglab1) %>%
        do.call(rbind,.)



## ------------------------------------------------------------------------
model_metrics

