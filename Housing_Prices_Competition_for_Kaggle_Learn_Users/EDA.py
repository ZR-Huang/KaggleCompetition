import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


data_path = "./data/train.csv"
data = pd.read_csv(data_path)

y = data['SalePrice']
X = data.drop(columns=['SalePrice'])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

low_cardinality_cate_cols = [cname for cname in X_train.columns if X_train[cname].nunique()<10 and X_train[cname].dtype == 'object']
high_cardinality_cate_cols = [cname for cname in X_train.columns if X_train[cname].nunique()>=10 and X_train[cname].dtype == 'object']

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

drop_X_train = X_train.drop(cols_with_missing, axis=1).select_dtypes(exclude='object')
drop_X_valid = X_valid.drop(cols_with_missing, axis=1).select_dtypes(exclude='object')


def get_scores_of_data(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictions)

print("The score of the dataset without any preprocessing:{}".format(
    get_scores_of_data(drop_X_train, drop_X_valid, y_train, y_valid)
)) # 17952.591404109586


imputer = SimpleImputer(strategy='most_frequent')
drop_X_train = X_train.select_dtypes(exclude='object')
drop_X_valid = X_valid.select_dtypes(exclude='object')
imputed_X_train = pd.DataFrame(imputer.fit_transform(drop_X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(drop_X_valid))

imputed_X_train.columns = drop_X_train.columns
imputed_X_valid.columns = drop_X_valid.columns

print("The score of the imputation dataset:{}".format(
    get_scores_of_data(imputed_X_train, imputed_X_valid, y_train, y_valid)
)) # 18290.918938356164

label_encoder = LabelEncoder()
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
for col in high_cardinality_cate_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_valid[col] = label_encoder.transform(X_valid[col])

# drop_X_train = X_train.drop(high_cardinality_cate_cols, axis=1)
# drop_X_valid = X_valid.drop(high_cardinality_cate_cols, axis=1)
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(imputed_X_train[low_cardinality_cate_cols]))
oh_cols_valid = pd.DataFrame(oh_encoder.transform(imputed_X_valid[low_cardinality_cate_cols]))
oh_cols_train.index = imputed_X_train.index
oh_cols_valid.index = imputed_X_valid.index

num_X_train = imputed_X_train.drop(low_cardinality_cate_cols, axis=1)
num_X_valid = imputed_X_valid.drop(low_cardinality_cate_cols, axis=1)

oh_X_train = pd.concat([num_X_train, oh_cols_train], axis=1)
oh_X_valid = pd.concat([num_X_valid, oh_cols_valid], axis=1)

print("The score of the imputation and OH dataset:{}".format(
    get_scores_of_data(oh_X_train, oh_X_valid, y_train, y_valid)
)) # 17555.522739726028





