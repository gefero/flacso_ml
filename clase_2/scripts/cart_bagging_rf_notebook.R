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
        index=cv_index,
        method="cv",
        number=5
        )


## ------------------------------------------------------------------------
grid_rf <- expand.grid(mtry=c(5,10,25),
                       min.node.size=c(5,20),
                       splitrule='gini'
        )


## ------------------------------------------------------------------------
t0 <- proc.time()
rf_tune <-  train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "ranger", 
                 trControl = fitControlrf,
                 tuneGrid = grid_rf,
                 metric='Accuracy'
)

proc.time() -  t0


## ----eval=FALSE, include=FALSE-------------------------------------------
## #saveRDS(rf_tune, '../models/rf_tune.RDS')


## ------------------------------------------------------------------------
rf_final <- train(imp_inglab1 ~ . , 
                 data = train, 
                 method = "ranger", 
                 tuneGrid = rf_tune$bestTune,
                 metric='Accuracy'
)


## ------------------------------------------------------------------------
y_preds_rf <- predict(rf_final, test)

confusionMatrix(y_preds_rf, test$imp_inglab1)



## ----eval=FALSE, include=FALSE-------------------------------------------
## 
## #saveRDS(rf_final, '../models/rf_final.RDS')

