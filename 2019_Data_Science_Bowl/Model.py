import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')
data.drop(['game_session', 'installation_id_x', 'installation_id_y', 'title', 'num_correct', 'num_incorrect', 'accuracy'], axis=1, inplace=True)

y = data['accuracy_group']
X_train = data.drop(['accuracy_group'], axis=1)

X_test = pd.read_csv('2019_Data_Science_Bowl/data/test_input.csv')
X_test.drop(['game_session', 'accuracy_group'], axis=1, inplace=True)

model = RandomForestRegressor(n_estimators=500, random_state=666)

model.fit(X_train, y)

pred = model.predict(X_test.drop(['installation_id'], axis=1))

for i in range(len(pred)):
    if pred[i] > 2.5:
        pred[i] = 3
    elif 2.5>= pred[i] > 1.5:
        pred[i] = 2
    elif 1.5>= pred[i] > 0.5:
        pred[i] = 1
    else:
        pred[i] = 0

output = pd.DataFrame({'installation_id': X_test.installation_id,
                      'accuracy_group': pred})

output.to_csv('2019_Data_Science_Bowl/data/submission.csv', index=False)