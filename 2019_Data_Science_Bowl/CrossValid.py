import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')
data.drop(['game_session', 'installation_id_x', 'installation_id_y', 'title', 'num_correct', 'num_incorrect', 'accuracy'], axis=1, inplace=True)

y = data['accuracy_group']
X = data.drop(['accuracy_group'], axis=1)

pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=500,
random_state=666))])

scores = -1*cross_val_score(pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_squared_error')

print("scores:\n", np.mean(scores))