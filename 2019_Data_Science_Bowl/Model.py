import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv('2019_Data_Science_Bowl/data/train_input.csv')

y = data['accuracy_group']
X_train = data.drop(['accuracy_group', 'session_title'], axis=1)

X_test = pd.read_csv('2019_Data_Science_Bowl/data/test_input.csv')
X_test.drop(['accuracy_group', 'session_title'], axis=1, inplace=True)


categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])


model = RandomForestRegressor(n_estimators=800, random_state=666)

model.fit(X_train, y)

pred = model.predict(X_test.drop(['installation_id'], axis=1))

for i in range(len(pred)):
    if pred[i] > 2.2:
        pred[i] = 3
    elif 2.2>= pred[i] > 1.7:
        pred[i] = 2
    elif 1.7>= pred[i] > 1:
        pred[i] = 1
    else:
        pred[i] = 0

submission = pd.read_csv('2019_Data_Science_Bowl/data/sample_submission.csv')

submission['accuracy_group'] = submission['accuracy_group'].astype('int')
submission.to_csv('2019_Data_Science_Bowl/data/submission.csv', index=False)