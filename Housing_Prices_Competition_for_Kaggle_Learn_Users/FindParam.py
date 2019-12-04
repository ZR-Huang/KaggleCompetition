from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from xgboost import XGBRegressor


train_data_path = './data/train.csv'

X = pd.read_csv(train_data_path)

y = X['SalePrice']
X.drop(['SalePrice'], axis=1, inplace=True)


numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

imputer = SimpleImputer(strategy='most_frequent')
imputed_X = pd.DataFrame(imputer.fit_transform(X.select_dtypes(exclude='object')))
imputed_X.columns = X.select_dtypes(exclude='object').columns


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
imputed_cat_cols = pd.DataFrame(imputer.fit_transform(X[categorical_cols]))
imputed_cat_cols.columns = X[categorical_cols].columns
oh_cols = pd.DataFrame(encoder.fit_transform(imputed_cat_cols))
oh_cols.index = X.index

oh_X = pd.concat([imputed_X, oh_cols], axis=1)


gs = GridSearchCV(XGBRegressor(objective="reg:squarederror"), 
                  {'n_estimators': range(100, 1000, 25), 
                   'learning_rate': [0.001, 0.003, 0.006, 0.009, 0.03, 0.06, 0.09, 0.1]}, 
                  cv=5,
                  scoring='neg_mean_absolute_error',
                  n_jobs=4,
                  verbose=1)

gs.fit(oh_X, y)

