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
##                  tuneGrid = grid_rf,
##                  metric='ROC'
## )
## proc.time() -  t0
## 
## 
## saveRDS(rf_fit_orig, '../models/rf_fit_orig.RDS')


## ----include=FALSE-------------------------------------------------------
rf_fit_orig <- readRDS('../models/rf_fit_orig.RDS')


## ------------------------------------------------------------------------
rf_fit_orig


## ------------------------------------------------------------------------
ggplot(train) + 
        geom_bar(aes(x=imp_inglab1))


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
rf_fit_up <- readRDS('../models/rf_fit_up.RDS')



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


## ------------------------------------------------------------------------
cart <- readRDS('../models/cart.RDS')


## ------------------------------------------------------------------------
model_list <- list(cart = cart,
                   rf_original = rf_fit_orig,
                   rf_weighted = rf_fit_wei,
                   rf_down = rf_fit_down,
                   rf_up = rf_fit_up)



## ------------------------------------------------------------------------
extract_conf_metrics <- function(model, data, obs){
        preds <- predict(model, data)        
        c<-confusionMatrix(preds, obs)
        results <- c(c$overall[1], c$byClass)
        return(results)
}



## ----eval=FALSE, include=TRUE--------------------------------------------
## 
## model_metrics <- model_list %>%
##         map(extract_conf_metrics, data=test, obs = test$imp_inglab1) %>%
##         do.call(rbind,.) %>%
##         as.data.frame()
## 


## ------------------------------------------------------------------------
model_metrics %>%
        select('Accuracy', 'Precision', 'Recall') %>%
        t()


## ------------------------------------------------------------------------
fitControlrf$seeds <- rf_fit_orig$control$seeds
#grid_ada <- expand.grid(mfinal=c(100,150), maxdepth=c(10,20), coeflearn='Breiman')
grid_ada <- expand.grid(nIter=c(100,150), method='Adaboost.M1')


## ----include=TRUE--------------------------------------------------------
t0 <- proc.time()
adaboost_fit_orig <-  train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "adaboost", 
                 trControl = fitControlrf,
                 tuneGrid = grid_ada,
                 metric='ROC'
)
proc.time() -  t0


saveRDS(adaboost_fit_orig, '../models/adaboost_fit_orig.RDS')



## ---- include=FALSE------------------------------------------------------
adaboost_fit_orig <- readRDS('../models/adaboost_fit_orig.RDS')


## ------------------------------------------------------------------------
adaboost_fit_orig


## ------------------------------------------------------------------------
dummy_pred <- function(y){
        uniqv <- unique(y)
        m<-uniqv[which.max(tabulate(match(y, uniqv)))]
        return(rep.int(m, length(y)))
}

eval_dummy <- function(obs){
        preds <- dummy_pred(obs)
        c <- confusionMatrix(preds, obs) 
        dummy <- c(c$overall[1], c$byClass)
        dummy <- as.data.frame(dummy) %>% t()
        return(dummy)
} 



## ------------------------------------------------------------------------
model_list <- list(cart = cart,
                   rf_original = rf_fit_orig,
                   rf_weighted = rf_fit_wei,
                   rf_down = rf_fit_down,
                   rf_up = rf_fit_up,
                   adaboost = adaboost_fit_orig)





## ------------------------------------------------------------------------
model_metrics <- model_list %>%
        map(extract_conf_metrics, data=test, obs = test$imp_inglab1) %>%
        do.call(rbind,.) %>%
        as.data.frame() %>%
        rbind(., eval_dummy(test$imp_inglab1)) %>%
        t()

model_metrics

