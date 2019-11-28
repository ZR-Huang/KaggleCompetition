import pandas as pd
from sklearn.ensemb

data_path = "./data/train.csv"
data = pd.read_csv(data_path)

y = data['SalePrice']
X = data.drop(columns=['SalePrice'])


s = (X.dtypes == 'object')
object_cols = list(s[s].index)

drop_X = X.select_dtypes(exclude=['object'])


