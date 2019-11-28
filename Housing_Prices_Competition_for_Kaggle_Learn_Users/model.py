import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


train_data_path = './data/train.csv'
test_data_path = './data/test.csv'

X = pd.read_csv(train_data_path)
X_test = pd.read_csv(train_data_path)

y = X['SalePrice']
X.drop(['SalePrice'], axis=1, inplace=True)

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
low_cardinality_cate_cols = [cname for cname in X_train.columns if X_train[cname].nunique()<10 and X_train[cname].dtype == 'object']
high_cardinality_cate_cols = [cname for cname in X_train.columns if X_train[cname].nunique()>=10 and X_train[cname].dtype == 'object']

numerical_transformer = SimpleImputer(strategy='most_frequent')

categorical_transformer 

# To be continue...
