booster=gbtree#gblinear for linear

scale_pos_weight = [] #use in case of unbalanced classes

#eta = [.005, .01, .015, .02]#learning rate <-Do after CV/RV

min_child_weight = [] #Helps with over/under fitting

subsample = [] #Subsample for each tree. Lower mean larger trees mean less overfitting

max_depth = []#maximum depth of tree - Also helps with over fitting

max_leaf_nodes = [] #max number of final nodes in a tree - I use max_depth instead

gamma = [] Need pos reduction in the loss function, gamma is the min loss red for split ->higher value makes model more conservative

alpha = [] # Also used for overfitting.L1 reg

lambda = []# Also used for overfitting. L2 reg


param_grid= {
'max_depth' : range(4,12,2),
'min_child_weight' : range(1,7,2),
'gamma' : range(0,.1,.5),
'subsample' : range(.6,1,.1),
'colsample_by_tree" :range(.6,1,.1)
}


from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, precision_score, recall_score
import xgboost as xgb



# 
xg = xgb.XGBClassifier(random_state=1985, eval_metric='map', n_jobs=-1, objective='binary:logistic')
#or logloss as metric.If its balanced (HA!) then auc is fine. Also  set scale_pos_weight (max ranking order) or max_delta_step=1 (maximizes correct prob) for inbalanced dataset
grid_xgb = GridSearchCV(xg,param_grid, cv=3, scoring='roc_auc')#also try ‘average_precision’, 'recall_score'
clf3.fit(X1, y1)

#pos_weight_2=y1[y1==0].count() / y1[y1==1].count()
#pos_weight = y1[y1==0].count()/y1.sum()
# If unbalanced
#'scale_pos_weight' :(pos_weight, pos_weight_2,1)







