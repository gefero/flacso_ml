## ------------------------------------------------------------------------
library(caret)
library(tidyverse)


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


## ------------------------------------------------------------------------
df_imp <- data %>%
        filter(imp_inglab1==1) %>%
        select(-imp_inglab1)

df_train <- data %>%
        filter(imp_inglab1==0) %>%
        select(-imp_inglab1) %>%
        mutate(p21 = case_when(
                        p21==0 ~ 100,
                        TRUE ~ p21))



## ------------------------------------------------------------------------
set.seed(957)


## ------------------------------------------------------------------------
cv_index <- createFolds(y = df_train$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)


## ------------------------------------------------------------------------
fitControl <- trainControl(
        index=cv_index,
        method="cv",
        number=5)


## ------------------------------------------------------------------------
lm_p21 <- train(p21 ~ ch04 + ch06, data = df_train, 
                 method = "lm", 
                 trControl = fitControl)

lm_p21


## ------------------------------------------------------------------------
lm_p21$finalModel


## ----warning=FALSE-------------------------------------------------------
lm_p21_b <- train(p21 ~ ., data = df_train, 
                 method = "lm", 
                 trControl = fitControl)


## ------------------------------------------------------------------------
lm_p21_b


## ------------------------------------------------------------------------
grid <- expand.grid(maxdepth=c(1, 2, 4, 8, 16))


## ----warning=FALSE-------------------------------------------------------
cart_p21 <- train(p21 ~ . , 
                 data = df_train, 
                 method = "rpart2", 
                 trControl = fitControl,
                 tuneGrid =grid)

cart_p21


## ----warning=TRUE--------------------------------------------------------
fitControl_rand <- trainControl(
        index=cv_index, 
        method="cv",
        number=5,
        search = 'random')


## ------------------------------------------------------------------------
cart_p21_rand <- train(p21 ~ ., data = df_train, 
                 method = "rpart2", 
                 trControl = fitControl_rand,
                 tuneLength = 2)

cart_p21_rand


## ------------------------------------------------------------------------
cart_p21


## ------------------------------------------------------------------------
saveRDS(cart_p21, '../models/p21_cart.rds')


## ------------------------------------------------------------------------
ggplot(cart_p21)


## ------------------------------------------------------------------------
cart_p21$bestTune


## ------------------------------------------------------------------------
set.seed(7412)
cv_index_final <- createFolds(y = df_train$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

fitControl_final <- trainControl(
        indexOut=cv_index_final, 
        method="cv",
        number=5)


## ------------------------------------------------------------------------
cart_final<-train(p21 ~ ., data = df_train,
                method = "rpart2", 
                trControl = fitControl_final, 
                tuneGrid = cart_p21$bestTune,
                metric='RMSE')

#saveRDS(rf_final, '../models/rf_final.RDS')

cart_final


## ------------------------------------------------------------------------
cart_final_f<-train(p21~., data=df_train,
                  method = "rpart2",
                  tuneGrid = cart_p21$bestTune)

cart_final_f


## ------------------------------------------------------------------------
y_preds_cart <- predict(cart_final_f, df_imp)


## ------------------------------------------------------------------------
preds <- cbind(y_preds_cart,
               df_imp$p21
)

colnames(preds) <- c('CART', 'Hot_Deck')

preds <- preds %>% as.data.frame() %>% gather(model, value)



## ------------------------------------------------------------------------
ggplot(preds) +
        geom_density(aes(x=value, fill=model), alpha=0.5)


## ------------------------------------------------------------------------
###

