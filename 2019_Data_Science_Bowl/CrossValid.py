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
X = data.drop(['accuracy_group', 'session_title'], axis=1)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])


param = {
  'n_estimators': range(200, 600, 25),
  'learning_rate': [0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06],
  'max_depth': [3, 6, 9, 12, 15],
  'min_child_weight': [1, 2, 3, 4, 5, 6, 7],
  'subsample' : [0.4, 0.5, 0.6, 0.7, 0.8],
}


gs = GridSearchCV(XGBRegressor(
                objective='reg:squarederror', random_state=666), 
                param_grid=param, 
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=15,
                verbose=1)

gs.fit(X, y)


print(gs.best_score_)
print(gs.best_params_)