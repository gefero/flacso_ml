## ---- message=FALSE----------------------------------------------------------
library(caret)
library(tidyverse)
library(rpart)


## ----------------------------------------------------------------------------
df <- MASS::Boston %>% mutate(chas=factor(chas, labels=c('No','Si')))
head(df)


## ----------------------------------------------------------------------------
set.seed(282)
tr_index <- createDataPartition(y=df$medv,
                                p=0.8,
                                list=FALSE)

train <- df[tr_index,]
test <- df[-tr_index,]

set.seed(655)
cv_index <- createFolds(y=train$medv,
                                k=5,
                                list=TRUE,
                                returnTrain=TRUE)


rf_trControl <- trainControl(
        index=cv_index,
        method="cv",
        number=5        
        )


rf_grid <- expand.grid(mtry=1:13,
                       min.node.size=c(5,10,15,20),
                       splitrule='variance'
                       )


## ----------------------------------------------------------------------------
t0 <- proc.time()
rf_fit <-  train(medv ~ . , 
                 data = train, 
                 method = "ranger", 
                 trControl = rf_trControl,
                 tuneGrid = rf_grid,
                 importance='impurity')
proc.time() -  t0


## ----------------------------------------------------------------------------

varimp <- function(data, y, model, loss='mse'){
        bool <- !names(data) %in% y
        X <- data[,bool]
        predic <- iml::Predictor$new(model, data=X, y=data[y])
        vi <- iml::FeatureImp$new(predic, loss='mse')
        return(vi)
}



## ----------------------------------------------------------------------------
ggpubr::ggarrange(
plot(varimp(data=train, y='medv', model=rf_fit)),
plot(varimp(data=test, y='medv', model=rf_fit, loss='mse')))


## ----------------------------------------------------------------------------
library(pdp)


## ----------------------------------------------------------------------------
partial(rf_fit, pred.var='rm')



## ----------------------------------------------------------------------------
partial(rf_fit, pred.var='rm', plot=TRUE, plot.engine='ggplot', rug=TRUE)


## ----------------------------------------------------------------------------
rf_fit %>%
        partial(pred.var='rm') %>%
        ggplot(aes(x=rm, y=yhat)) +
                geom_line() +
                geom_smooth(se=FALSE) 


## ----------------------------------------------------------------------------
pd <- partial(rf_fit, pred.var=c('rm', 'lstat'))
plotPartial(pd)


## ----------------------------------------------------------------------------
partial(rf_fit, pred.var='chas') %>%
        ggplot(aes(x=chas, y=yhat)) +
                geom_bar(stat='identity')


## ----------------------------------------------------------------------------
ggpubr::ggarrange(
        partial(rf_fit, pred.var='rm', plot=TRUE, ice=TRUE, rug=TRUE, 
        plot.engine = 'ggplot', alpha=0.1),
        partial(rf_fit, pred.var='lstat', plot=TRUE, ice=TRUE, rug=TRUE, 
        plot.engine = 'ggplot', alpha=0.1)
)

