booster=gbtree#gblinear for linear

scale_pos_weight = [] #use in case of unbalanced classes

#eta = [.005, .01, .015, .02]#learning rate <-Do after CV/RV

min_child_weight = [] #Helps with over/under fitting

subsample = [] #Subsample for each tree. Lower mean larger trees mean less overfitting

max_depth = []#maximum depth of tree - Also helps with over fitting

max_leaf_nodes = [] #max number of final nodes in a tree - I use max_depth instead

gamma = [] #Need pos reduction in the loss function, gamma is the min loss red for split ->higher value makes model more conservative

alpha = [] # Also used for overfitting.L1 reg

lambda = []# Also used for overfitting. L2 reg

#pos_weight_2=y1[y1==0].count() / y1[y1==1].count()
#pos_weight = y1[y1==0].count()/y1.sum()
# If unbalanced
#'scale_pos_weight' :(pos_weight, pos_weight_2,1)

param_grid= {
'max_depth' : range(4,12,2)
,'min_child_weight' : range(1,7,2)
,'gamma':[i/100.0 for i in range(0,10)]
,'subsample':[i/10.0 for i in range(4,10)]
,'colsample_bytree':[i/10.0 for i in range(4,10)]
}
#NOTE - 1500 total combinations

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, precision_score, recall_score
import xgboost as xgb



# 
xg = xgb.XGBClassifier(random_state=1985, eval_metric='map', n_jobs=-1, objective='binary:logistic')

#or logloss as metric.If its balanced (HA!) then auc is fine. Also  set scale_pos_weight (max ranking order) or max_delta_step=1 (maximizes correct prob) for inbalanced dataset
grid_xgb = GridSearchCV(xg, param_grid = param_grid, cv=3, scoring='roc_auc')#also try ‘average_precision’, 'recall_score'

grid_xgb.fit(train.drop(columns=[target]), train[target])

grid_xgb.cv_results_
grid_xgb.best_estimator_
grid_xgb.best_score_
grid_xgb.best_params_

n_iter_search = 50
ran_xgb = RandomizedSearchCV(xg, param_distributions=param_grid, cv=3, scoring='roc_auc', n_iter=n_iter_search,)

ran_xgb.fit(train.drop(columns=[target]), train[target])

ran_xgb.cv_results_
ran_xgb.best_estimator_
ran_xgb.best_score_
ran_xgb.best_params_






