import pandas as pd
from concurrent.futures import ProcessPoolExecutor


class Preprocess:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns
    
    def re_organize(self):
        # create the new DataFrame with the new columns
        # cols = self.get_columns()
        new_df = pd.DataFrame(columns=self.columns)

        # calculate the new data for the new DF
        record = self.new_record(self.columns)
        total_time = 0
        total_event_count = 0
        max_time_of_one_activity = 0
        max_event_count_of_one_activity = 1
        for row in self.data.itertuples(index=True):
            event_code = getattr(row, 'event_code')
            _type = getattr(row, 'type')
            
            if event_code == 2000:
                total_time += max_time_of_one_activity
                total_event_count += max_event_count_of_one_activity
                max_time_of_one_activity = 0
                max_event_count_of_one_activity = 1
                if _type == 'Assessment':
                    # append the new data into the new DF
                    record['game_session'] = getattr(row, 'game_session')
                    record['installation_id'] = getattr(row, 'installation_id')
                    record['event_count'] = total_event_count
                    record['game_time'] = total_time
                    new_df.loc[new_df.__len__()] = list(record.values())

                record[getattr(row, 'title')] += 1
                record[getattr(row, 'type')] += 1
                record[getattr(row, 'world')] += 1
            
            max_time_of_one_activity = getattr(row, 'game_time')
            max_event_count_of_one_activity = getattr(row, 'event_count')
            record[event_code] += 1

        return new_df
        
    def get_columns(self):
        try:
            event_codes = self.data['event_code'].unique()
            titles = self.data['title'].unique()
            types = self.data['type'].unique()
            world = self.data['world'].unique()
        except TypeError:
            print(self.data)
            import sys
            sys.exit(0)
        
        self.columns.extend(['game_session', 'installation_id', 'event_count', 'game_time'])
        self.columns.extend(event_codes)
        self.columns.extend(titles)
        self.columns.extend(types)
        self.columns.extend(world)

    def new_record(self, keys):
        d = dict()
        for key in keys:
            d[key] = 0
        
        return d


def task(packed_parameters):
    # packed_parameters = (columns, data_of_one_id)
    columns, data, test = packed_parameters
    p = Preprocess(data[1], columns) # here data is a DataFrameGroupby object which is a tuple (group name, df)
    if test:
        return p.re_organize().tail(1)
    return p.re_organize()


def process_data(filepath, labelpath, targetpath, test=False):
    # train = pd.read_csv('2019_Data_Science_Bowl/data/train.csv')
    # train_labels = pd.read_csv('2019_Data_Science_Bowl/data/train_labels.csv')
    # test = pd.read_csv('2019_Data_Science_Bowl/data/test.csv')
    # sample_submission = pd.read_csv('2019_Data_Science_Bowl/data/sample_submission.csv')
    data = pd.read_csv(filepath)
    label = pd.read_csv(labelpath)
    # get the new columns for the new data
    columns = []
    event_codes = data['event_code'].unique()
    titles = data['title'].unique()
    types = data['type'].unique()
    world = data['world'].unique()
    columns.extend(['game_session', 'installation_id', 'event_count', 'game_time'])
    columns.extend(event_codes)
    columns.extend(titles)
    columns.extend(types)
    columns.extend(world)

    groups = data.groupby('installation_id')
        
    packed_parameters = [(columns, group, test) for group in groups]
    
    with ProcessPoolExecutor() as executor:
        data = pd.concat([df for df in executor.map(task, packed_parameters)], ignore_index=True)
    
    if test:
        # print(data)
        data = pd.merge(label, data, how='inner', on='installation_id')
    else:
        data = pd.merge(label, data, how='inner', on='game_session')

    data.to_csv(targetpath, index=False)

if __name__ == "__main__":
    # process_data("2019_Data_Science_Bowl/data/train.csv",
    # "2019_Data_Science_Bowl/data/test_labels.csv",
    # "2019_Data_Science_Bowl/data/train_input.csv")
    process_data("2019_Data_Science_Bowl/data/test.csv", 
    "2019_Data_Science_Bowl/data/sample_submission.csv", 
    "2019_Data_Science_Bowl/data/test_input.csv", test=True)