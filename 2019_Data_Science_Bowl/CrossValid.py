import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')
y = data['accuracy_group']
X = data.drop(['accuracy_group', 'session_title'], axis=1)
# data.drop([], axis=1, inplace=True)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# pipeline = Pipeline(steps=[
#     ('model', RandomForestRegressor(n_estimators=500,random_state=666)),
#     ])

gs = GridSearchCV(RandomForestRegressor(random_state=666), 
                  {'n_estimators': range(100, 1000, 25), 
                    }, 
                  cv=3,
                  scoring='neg_root_mean_squared_error',
                  n_jobs=10,
                  verbose=1)

# scores = -1*cross_val_score(pipeline, X, y,
#                               cv=5,
#                               scoring='neg_mean_squared_error',
#                               n_jobs=4,
#                               verbose=1)

# print("scores:\n", np.mean(scores))

gs.fit(X, y)
print(gs.best_score_)
print(gs.best_params_)