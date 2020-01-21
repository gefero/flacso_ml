## ----message=FALSE, warning=FALSE----------------------------------------
library(tidyverse)
library(rpart)
library(caret)

set.seed(42)
X <- runif(100, 0, 1) - 0.5
y <- 3*X**2 + 0.05 * runif(100,0,4)

df <- cbind(X,y) %>% as_tibble()

rm(X,y)



## ------------------------------------------------------------------------
tree_1 <- rpart(y~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                

df <- df %>% mutate(h_1 = predict(tree_1, df),
                p_1 = h_1,
                y_1 = y - p_1)



## ------------------------------------------------------------------------
tree_2 <- rpart(y_1~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                
df <- df %>% mutate(h_2 = predict(tree_2, df),
                    p_2 = p_1 + h_2,
                    y_2 = y - p_2)


## ------------------------------------------------------------------------
tree_3 <- rpart(y_2~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                

df <- df %>% mutate(h_3 = predict(tree_3, df),
                    p_3 = p_2 + h_3,
                    y_3 = y - p_3)

tree_4 <- rpart(y_3~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                

df <- df %>% mutate(h_4 = predict(tree_4, df),
                    p_4 = p_3 + h_4,
                    y_4 = y - p_4)

tree_5 <- rpart(y_4~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                

df <- df %>% mutate(h_5 = predict(tree_5, df),
                    p_5 = p_4 + h_5,
                    y_5 = y - p_5)

tree_6 <- rpart(y_5~X, data=df, method='anova', control=list(cp=0.0000001, maxdepth=2))                

df <- df %>% mutate(h_6 = predict(tree_6, df),
                    p_6 = p_5 + h_6,
                    y_6 = y - p_6)



## ----fig.height=15, fig.width=10-----------------------------------------
ggpubr::ggarrange(ncol=2, nrow=3,
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y), color='blue') + 
                          geom_line(aes(x=X, y=h_1), color='green'),
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y), color='blue') + 
                          geom_line(aes(x=X, y=p_1), color='red'),
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y_1), color='blue') + 
                          geom_line(aes(x=X, y=h_2), color='green') +
                          scale_y_continuous(limits=c(-0.4,1)),
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y), color='blue') + 
                          geom_line(aes(x=X, y=p_2), color='red'),
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y_2), color='blue') + 
                          geom_line(aes(x=X, y=h_3), color='green') +
                          scale_y_continuous(limits=c(-0.4,1)),
                  ggplot(df) + 
                          geom_point(aes(x=X, y=y), color='blue') + 
                          geom_line(aes(x=X, y=p_3), color='red')
)


## ------------------------------------------------------------------------
trainControl <- trainControl(method='none')

gbm <- train(y~X, method='gbm',
             data=df,
             trControl=trainControl,
             tuneGrid=data.frame(n.trees=3,
                                 interaction.depth=3,
                                 shrinkage=1.0,
                                 n.minobsinnode=2)
)


gbm_n3_s01 <- train(y~X, method='gbm',
             data=df,
             trControl=trainControl,
             tuneGrid=data.frame(n.trees=3,
                                 interaction.depth=3,
                                 shrinkage=0.1,
                                 n.minobsinnode=2)
             )

gbm_n200_s1 <- train(y~X, method='gbm',
             data=df,
             trControl=trainControl,
             tuneGrid=data.frame(n.trees=200,
                                 interaction.depth=3,
                                 shrinkage=1,
                                 n.minobsinnode=2)
             )

gbm_slow <- train(y~X, method='gbm',
             data=df,
             trControl=trainControl,
             tuneGrid=data.frame(n.trees=200,
                                 interaction.depth=3,
                                 shrinkage=0.1,
                                 n.minobsinnode=2)
             )




## ----fig.height=10, fig.width=10-----------------------------------------
df <- df %>% mutate(y_gbm=predict(gbm, df),
                    y_gbm_n3_s01=predict(gbm_n3_s01, df),
                    y_gbm_n200_s1=predict(gbm_n200_s1, df),
                    y_gbm_slow=predict(gbm_slow, df)
                    )

ggpubr::ggarrange(
ggplot(df) + 
        geom_point(aes(x=X, y=y), color='blue') + 
        geom_line(aes(x=X,y=y_gbm)) +
        labs(title='n.trees=3, shrinkage=1'),

ggplot(df) + 
        geom_point(aes(x=X, y=y), color='blue') + 
        geom_line(aes(x=X,y=y_gbm_n3_s01)) +
        labs(title='n.trees=3, shrinkage=0.1'),


ggplot(df) + 
        geom_point(aes(x=X, y=y), color='blue') + 
        geom_line(aes(x=X,y=y_gbm_n200_s1)) +
        labs(title='n.trees=200, shrinkage=1'),

ggplot(df) + 
        geom_point(aes(x=X, y=y), color='blue') + 
        geom_line(aes(x=X,y=y_gbm_slow)) +
        labs(title='n.trees=200, shrinkage=0.1')
)

