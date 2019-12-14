import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score

from xgboost import XGBRegressor

from functools import partial
import scipy

import multiprocessing
n_jobs = multiprocessing.cpu_count()-1


data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')
y = data['accuracy_group']
X = data.drop(['accuracy_group'], axis=1)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])


param = {
  'n_estimators': range(150, 400, 25),
  'learning_rate': [0.006, 0.009, 0.01, 0.03, 0.06, 0.09],
  'max_depth': [7, 8, 9, 10, 11],
}

# param = {
#   'n_estimators': [275],
#   'learning_rate': [0.06],
#   'max_depth': [9],
# }
# Best score: -1.135999577937502

param = {
  'n_estimators': [150],
  'learning_rate': [0.03],
  'max_depth': [7],
}
# Best Score : -1.068639288246002


gs = GridSearchCV(XGBRegressor(
                objective='reg:squarederror', random_state=666), 
                param_grid=param, 
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=n_jobs,
                verbose=1)

gs.fit(X, y)


print(gs.best_score_)
print(gs.best_params_)