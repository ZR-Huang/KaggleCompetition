import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from collections import Counter
from tqdm import tqdm

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


if __name__ == "__main__":
    # p_train = Preprocess('2019_Data_Science_B')
    p_test = Preprocess('2019_Data_Science_Bowl/data/test.csv')
    p_test.get_data(test_set=True).to_csv('2019_Data_Science_Bowl/data/test_input.csv', index=False)