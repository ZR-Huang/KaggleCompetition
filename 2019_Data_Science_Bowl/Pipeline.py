import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score, accuracy_score

import xgboost as xgb

from functools import partial
import scipy

# create by the kaggle environment
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


Kaggle = False
if Kaggle:
    DIR = '/kaggle/input/data-science-bowl-2019/'
else:
    DIR = '2019_Data_Science_Bowl/data/'


class Preprocess:

    def __init__(self, filepath):

        self.filepath = filepath
        # self.data = self.load_data
        # self.columns = columns

    def __load_data(self):
        self.data = pd.read_csv(self.filepath)
        # return pd.read_csv(self.filepath)
    
    def __get_title_list(self):
        self.title_list = list(self.data.title.unique())
        # return list(self.data.title.unique())

    def __get_event_code_list(self):
        self.event_code_list = list(self.data.event_code.unique())
        # return list(self.data.event_code.unique())

    def __get_win_code_of_title(self):
        self.win_code_of_title = dict(zip(self.title_list, (np.ones(len(self.title_list))).astype('int')* 4100))
        self.win_code_of_title['Bird Measurer (Assessment)'] = 4110
    
    def compile_data(self, user_sample, test_set = False):
        '''
        user_sample : DataFrame from train/test group by 'installation_id'
        test_set : related with the labels processing
        '''

        # Constants and parameters declaration
        user_assessments = []
        last_type = None
        types_count = {'Clip': 0, 'Activity': 0, 'Assessment':0, 'Game':0}
        time_first_activity = float(user_sample['timestamp'].values[0])
        time_spent_each_title = {title: 0 for title in self.title_list}
        event_code_count = {code: 0 for code in self.event_code_list}
        accuracy_groups_nums = {0: 0, 1:0, 2:0, 3:0}

        accumu_accuracy_group = 0
        accumu_accuracy = 0
        accumu_win_n = 0
        accumu_loss_n = 0
        accumu_actions = 0
        counter = 0
        durations = []

        # group by 'game_session'
        for i, session in user_sample.groupby('game_session', sort=False):
            # i : game_session_id
            # session : DataFrame from user_sample group by 'game_session'
            session_type = session['type'].iloc[0]
            session_title = session['title'].iloc[0]

            if session_type != 'Assessment':
                time_spent_each_title[session_title] += session['game_time'].iloc[-1]
            
            if session_type == 'Assessment' and (test_set or len(session) > 1):
                # search for event_code 4100 (4110)
                all_4100 = session.query(f'event_code == {self.win_code_of_title[session_title]}')
                # numbers of wins and losses
                win_n = all_4100['event_data'].str.contains('true').sum()
                loss_n = all_4100['event_data'].str.contains('false').sum()

                # initialize features and then update
                features = deepcopy(types_count)
                features.update(deepcopy(time_spent_each_title))
                features.update(deepcopy(event_code_count))
                features['session_title'] = session_type
                features['accumu_win_n'] = accumu_win_n
                features['accumu_loss_n'] = accumu_loss_n
                accumu_win_n += win_n
                accumu_loss_n += loss_n

                features['day_of_the_week'] = (session['timestamp'].iloc[-1]).strftime('%A')

                if durations == []:
                    features['duration_mean'] = 0
                else:
                    features['duration_mean'] = np.mean(durations)
                durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

                # average of the all accuracy of this player
                features['accuracy_ave'] = accumu_accuracy / counter if counter > 0 else 0
                accuracy = win_n / (win_n + loss_n) if (win_n + loss_n) > 0 else 0
                accumu_accuracy += accuracy

                if accuracy == 0:
                    features['accuracy_group'] = 0
                elif accuracy == 1:
                    features['accuracy_group'] = 3
                elif accuracy == 0.5:
                    features['accuracy_group'] = 2
                else:
                    features['accuracy_group'] = 1
                features.update(accuracy_groups_nums)
                accuracy_groups_nums[features['accuracy_group']] += 1

                # average of accuray_groups of this player
                features['accuracy_group_ave'] = accumu_accuracy_group / counter if counter > 0 else 0
                accumu_accuracy_group += features['accuracy_group']

                # how many actions the player has done before this Assessment
                features['accumu_actions'] = accumu_actions

                # if test_set, all session belong th the final assessment
                # elif train, needs to be passed through this piece of code
                if test_set or (win_n + loss_n) > 0:
                    user_assessments.append(features)
                
                counter  += 1
        
            # how many actions was made in each event_code
            event_codes = Counter(session['event_code'])
            for key in event_codes.keys():
                event_code_count[key] += event_codes[key]

            # how many actions the player has done
            accumu_actions += len(session)
            if last_type != session_type:
                types_count[session_type] += 1
                last_type = session_type

        # if test_set, only the lase assessment must be predicted,
        # the previous are scraped
        if test_set:
            return user_assessments[-1]
        return user_assessments

    def get_data(self, test_set=False):
        # get things done
        self.__load_data()
        self.__get_title_list()
        self.__get_event_code_list()
        self.__get_win_code_of_title()
        # convert 'timestamp' to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        complied_data = []
        installation_n = self.data['installation_id'].nunique()
        if test_set:
            for _, user_sample in tqdm(self.data.groupby('installation_id', sort=False), total=installation_n):
                complied_data.append(self.compile_data(user_sample, test_set=test_set))
        else:
            for _, user_sample in tqdm(self.data.groupby('installation_id', sort=False), total=installation_n):
                complied_data += self.compile_data(user_sample, test_set=test_set)
        return pd.DataFrame(complied_data)


# Load the train/test data
p_train = Preprocess(os.path.join(DIR, 'train.csv'))
p_test = Preprocess(os.path.join(DIR, 'test.csv'))

data = p_train.get_data()
y = data['accuracy_group']
X_train = data.drop(['accuracy_group', 'session_title'], axis=1)

X_test = p_test.get_data(test_set=True)
X_test.drop(['accuracy_group', 'session_title'], axis=1, inplace=True)


categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

# Start training models
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=666)

xgb_models = []
scores = []
params = {
    'max_depth' : 9,
    'learning_rate': 0.06,
    'n_estimators': 275,
    'objective': 'reg:squarederror',
    'seed' : 666
}

for fold, (train_ids, val_ids) in enumerate(folds.split(X_train, y)):
    print(f'- Fold :{fold+1} / {NFOLDS}')
    dtrain = xgb.DMatrix(X_train.iloc[train_ids], y[train_ids])
    dval = xgb.DMatrix(X_train.iloc[val_ids], y[val_ids])
    model = xgb.train(params = params,
                      dtrain = dtrain,
                      num_boost_round=5000,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=100,
                      verbose_eval=50)
    xgb_models.append(model)

# Predict each Model
preds = []
for model in xgb_models:
    pred = model.predict(xgb.DMatrix(X_train))
    pred = pred.flatten()
    preds.append(pred)

df = pd.DataFrame(preds).T
df.columns = ['X1', 'X2', 'X3', 'X4', 'X5']
df['mean'] = df.mean(axis='columns')

print(df.head(10))


class OptRounder:
    def __init__(self):
        self.res_ = []
        self.coef_ = []
    
    def get_res(self):
        return self.res_

    def func(self, coef, X, y):
        # objective function
        kappa = cohen_kappa_score(self.bincut(coef, X), y, weights='quadratic')
        return -kappa
    
    def bincut(self, coef, X):
        return pd.cut(X,
                    [-np.inf]+list(np.sort(coef))+[np.inf],
                    labels= [0, 1, 2, 3])
    
    def fit(self, X, y):
        pfunc = partial(self.func, X=X, y=y)
        self.res_ = scipy.optimize.minimize(fun=pfunc, # objective func
                                            x0 = [0.6, 1.5, 2.4], # initial coefficients
                                            method='nelder-mead')   # solver
        self.coef_ = self.res_.x
    
    def predict(self, X, coef):
        return self.bincut(coef, X)

# Optimize Rounding Coefficients
optR = OptRounder()
optR.fit(df['mean'].values.reshape(-1,), y)
res = optR.get_res()
coefficients = res.x
print('- Iterations performed\t:{res.nit}')
print('- Optimized coefficients\t:{coefficients}')
print(f'- Cohen Kappa Score\t:{-res.fun}')

# final classification
df['predict'] = optR.predict(df['mean'].values, coefficients).astype(int)
df['y'] = y
acc = accuracy_score(df['y'], df['predict'])
print('- Accuracy of the final classification\t:{acc}')

X_test = X_test[list(X_train.columns)]
# Make submission
preds = []
for model in xgb_models:
    pred = model.predict(xgb.DMatrix(X_test))
    pred = pred.flatten()
    preds.append(pred)

df_submission = pd.DataFrame(preds).T
df_submission['mean'] = df_submission.mean(axis='columns')
df_submission['pred'] = optR.predict(df_submission['mean'].values, coefficients).astype(int)

submission = pd.read_csv(os.path.join(DIR, 'sample_submission.csv'))
submission['accuracy_group'] = df_submission['pred']
submission.to_csv('submission.csv', index=False)