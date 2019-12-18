import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

from functools import partial
import scipy

def qwk(act,pred,n=4,hist_range=(0,3)):
    
    O = confusion_matrix(act,pred)
    O = np.divide(O,np.sum(O))
    
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E))
    
    num = np.sum(np.multiply(W,O))
    den = np.sum(np.multiply(W,E))
        
    return 1-np.divide(num,den)

data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')
y = data['accuracy_group']
X_train = data.drop(['accuracy_group'], axis=1)

all_feature = [cname for cname in X_train.columns]
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])

# Start training models
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=666)

xgb_models = []
xgb_params = {
    'max_depth' : 7, # 5->7
    'learning_rate': 0.03, # 0.029 -> 0.03
    'n_estimators': 150, # 240 -> 150
    'objective': 'reg:squarederror',
    # 'min_child_weight': 0.9,
    # 'subsample': 0.45,
    'seed' : 666,
}

for fold, (train_ids, val_ids) in enumerate(folds.split(X_train, y)):
    print(f'- Fold :{fold+1} / {NFOLDS}')
    dtrain = xgb.DMatrix(X_train.iloc[train_ids], y[train_ids])
    dval = xgb.DMatrix(X_train.iloc[val_ids], y[val_ids])
    model = xgb.train(params = xgb_params,
                      dtrain = dtrain,
                      num_boost_round=5000,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=100,
                      verbose_eval=50)
    xgb_models.append(model)
    

cat_models = []
def get_catboost():
    return CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        random_seed=666,
        depth=10,
        border_count=108,
        bagging_temperature=2.348502,
        early_stopping_rounds=200
    )

for fold, (train_ids, val_ids) in enumerate(folds.split(X_train, y)):
    print(f'- Fold :{fold+1} / {NFOLDS}')
    model = get_catboost()
    model.fit(X_train.loc[train_ids, all_feature], y.loc[train_ids],
    eval_set=(X_train.loc[val_ids, all_feature], y.loc[val_ids]),
    use_best_model=False,
    verbose=500,
    cat_features=categorical_cols)
    cat_models.append(model)


lgb_models = []
lgb_params = {
    'n_jobs': -1,
    'seed': 666,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'subsample': 0.75,
    'feature_fraction':0.998495,    # add
    'bagging_fraction': 0.872417,   # mod 0.8→
    'bagging_freq': 1,              # add
    'colsample_bytree': 0.8,        # add
    'subsample_freq': 1,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'max_depth': 13,                # mod 10→
    'num_leaves': 1028,             # mod      # 2^max_depth < num_leaves
    'min_gain_to_split':0.085502,   # add
    'min_child_weight':1.087712,    # add
    'lambda_l1': 1,  
    'lambda_l2': 1,
    'verbose': 100,
}
orginal_X_train_cols = X_train.columns
X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
for fold, (train_ids, val_ids) in enumerate(folds.split(X_train, y)):
    print(f'- Fold :{fold+1} / {NFOLDS}')
    train_set = lgb.Dataset(X_train.iloc[train_ids], y[train_ids], categorical_feature=categorical_cols)
    val_set = lgb.Dataset(X_train.iloc[val_ids], y[val_ids], categorical_feature=categorical_cols)
    model = lgb.train(params=lgb_params, train_set=train_set, valid_sets=[train_set, val_set],
    num_boost_round=5000, early_stopping_rounds=100, verbose_eval=100)
    lgb_models.append(model)

X_train.columns = orginal_X_train_cols

# Predict each Model
preds = []
for model in xgb_models:
    pred = model.predict(xgb.DMatrix(X_train))
    pred = pred.flatten()
    preds.append(pred)
for model in cat_models:
    pred = model.predict(X_train)
    preds.append(pred)
for model in lgb_models:
    pred = model.predict(X_train, num_iteration=model.best_iteration)
    pred = pred.reshape(len(X_train), 1).flatten()
    preds.append(pred)

df = pd.DataFrame(preds).T
df.columns = [
    'X1', 'X2', 
    'X3', 'X4', 'X5',
    'C1', 'C2', 
    'C3', 'C4', 'C5',
    'L1', 'L2', 
    'L3', 'L4', 'L5',
    ]
df['mean'] = df.mean(axis='columns')
df['X_mean'] = df[['X1', 'X2', 'X3', 'X4', 'X5']].mean(axis='columns')
df['C_mean'] = df[['C1', 'C2', 'C3', 'C4', 'C5']].mean(axis='columns')
df['L_mean'] = df[['L1', 'L2', 'L3', 'L4', 'L5']].mean(axis='columns')
print(df.head(10))


class OptRounder:
    def __init__(self):
        self.res_ = []
        self.coef_ = []
    
    def get_res(self):
        return self.res_

    def func(self, coef, X, y):
        # objective function
        kappa = cohen_kappa_score(self.bincut(coef, X), y, weights='quadratic')
        return -kappa
    
    def bincut(self, coef, X):
        return pd.cut(X,
                    [-np.inf]+list(np.sort(coef))+[np.inf],
                    labels= [0, 1, 2, 3])
    
    def fit(self, X, y):
        pfunc = partial(self.func, X=X, y=y)
        self.res_ = scipy.optimize.minimize(fun=pfunc, # objective func
                                            x0 = [0.6, 1.5, 2.4], # initial coefficients
                                            method='nelder-mead')   # solver
        self.coef_ = self.res_.x
    
    def predict(self, X, coef):
        return self.bincut(coef, X)

# Optimize Rounding Coefficients
for col in ['X_mean', 'C_mean', 'L_mean']:
    optR = OptRounder()
    optR.fit(df[col].values.reshape(-1,), y)
    res = optR.get_res()
    coefficients = res.x
    print(f'- Iterations performed\t:{res.nit}')
    print(f'- Optimized coefficients\t:{coefficients}')
    print(f'- Cohen Kappa Score\t:{-res.fun}')
    # final classification
    df[col.replace('_mean', '_pred')] = optR.predict(df[col].values, coefficients).astype(int)

optR = OptRounder()
optR.fit(df['mean'].values.reshape(-1,), y)
res = optR.get_res()
coefficients = res.x
print(f'- Iterations performed\t:{res.nit}')
print(f'- Optimized coefficients\t:{coefficients}')
print(f'- Cohen Kappa Score\t:{-res.fun}')
df['all_pred'] = optR.predict(df['mean'].values, coefficients).astype(int)

# Cohen Kappa Score [X_mean]: 0.71471 -> 0.79827 after modifying the params
# Cohen Kappa Score [C_mean]: 0.78973
# Cohen Kappa Score [L_mean]: 0.84958
# Cohen Kappa Score [all_mean] : 0.79424 -> 0.81785 because of the score of the X_mean increasing

print(df.head(20))

# df.to_csv('2019_Data_Science_Bowl/data/pred.csv')

# df = pd.read_csv('2019_Data_Science_Bowl/data/pred.csv').drop(['Unnamed: 0'], axis=1)
# print(df)
result = df[['X_pred', 'C_pred', 'L_pred', 'all_pred']].mode(axis='columns')
# print(result.head(20))
for index, row in result.iterrows():
    if not pd.isnull(row[1]):
        result.iloc[index][0] = df.iloc[index]['L_pred'] # use all_pred QWK: 0.79420


result.drop(columns=[1], inplace=True)
result = result.astype(int)
# print(result.head(20))

# df['final_pred'] = df[['X_pred', 'C_pred', 'L_pred', 'all_pred']].mode(axis='columns').astype(int)

print('OOF QWK:', qwk(y, result))
print("Accuray:", accuracy_score(y, result))

