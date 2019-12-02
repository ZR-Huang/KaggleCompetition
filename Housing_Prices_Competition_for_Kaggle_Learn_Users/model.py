import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV


train_data_path = './data/train.csv'
test_data_path = './data/test.csv'

X = pd.read_csv(train_data_path)
X_test = pd.read_csv(test_data_path)


y = X['SalePrice']
X.drop(['SalePrice'], axis=1, inplace=True)

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

numerical_transformer = SimpleImputer(strategy='most_frequent')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cate', categorical_transformer, categorical_cols),
    ]
)


model = XGBRegressor(objective="reg:squarederror")

gs = GridSearchCV(model, 
                  {'n_estimators': range(100, 1000, 25), 
                   'learning_rate': [0.001, 0.003, 0.006, 0.009, 0.03, 0.06, 0.09, 0.1]}, 
                  cv=5,
                  scoring='reg_mean_absolute_error',
                  n_jobs=4,
                  verbose=1)

gs_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gs', gs)
])

print(gs.best_params_)
print(gs.best_score_)
"""
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("Scores:{} | Mean score:{}".format(scores, scores.mean()))
# 15658

pipeline.fit(X, y)
preds_test = pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.Id,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
"""