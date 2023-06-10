import os
import pandas as pd
import numpy as np 
import datetime as dt

from copy import deepcopy

class Preprocessor():
    def process(self, args, raw_data_dir, cleaned_data_dir):
        
        min_len = args.min_len
        min_count = args.min_count
        self.data_name = args.data_name.lower()
        self.set_keys()
        
        step = 1;
        print("step{}: Loading data".format(step))
        data = self.load_data(raw_data_dir)
                
        step += 1
        print("step{}: Adding time stamp".format(step))
        data = self.add_time_feature(data)
                
        step += 1
        print("step{}: Deleting rare items and short sessions".format(step))
        data = self.remove_short_sessions(data, min_len)
        data = self.remove_rare_items(data, min_count)
        data = self.remove_short_sessions(data, min_len)
                
        step += 1
        print("step{}: Splitting last day".format(step))
        data, test_data = self.get_test_data(data)
        data = self.remove_short_sessions(data, min_len)     
               
        step += 1
        print("step{}: Removing non-common items from test set".format(step))
        test_data = self.remove_non_common_items(data, test_data)
        test_data = self.remove_short_sessions(test_data, min_len)
    
        step += 1
        print("step{}: Adding item idx".format(step))
        data , test_data = self.add_item_idx(data, test_data)

        step += 1
        print("step{}: Adding Time Intervals".format(step))    
        data, test_data = self.add_delta_t(data, test_data)
             
        step += 1
        print("step{}: Saving datasets".format(step))
        
        self.print_stat(test_data, (self.data_name + " test"))
        self.save(test_data, cleaned_data_dir, "test.pkl")
        
        self.print_stat(data, (self.data_name + " train"))
        self.save(data, cleaned_data_dir, "train_1_1.pkl")

        if(self.data_name == "yoochoose"):
                            
            train_1_4, train_1_64 = self.fraction(data)
            
            self.print_stat(train_1_64, (self.data_name + " train_1_64"))
            self.save(train_1_64, cleaned_data_dir, "train_1_64.pkl")
            
            self.print_stat(train_1_4, (self.data_name + " train_1_4"))
            self.save(train_1_4, cleaned_data_dir, "train_1_4.pkl")
        
    def set_keys(self):        
        self.session_key = "session_id"
        self.item_key = "item_id"
        
        if(self.data_name == "yoochoose"):        
            self.time_key = "time_str"
            self.cat_key = "category"
        else :
            self.time_key = "eventdate"
    
    def add_item_idx(self, data, test_data):
        item_ids = data[self.item_key].unique()

        first_index = 0
        last_index = first_index + len(item_ids)
        item_map = pd.DataFrame({"item_idx":np.arange(first_index, last_index),
                                  self.item_key:item_ids});
        
        data = pd.merge(data, item_map, on=self.item_key, how='inner')
        test_data = pd.merge(test_data, item_map, on=self.item_key, how='inner')
        return data, test_data
    
    def add_delta_t(self, train_data, test_data):
        len_test = len(test_data)
        len_train = len(train_data)

        if(len(train_data) != len_train and len(test_data) != len_test):
            raise
        
        if(self.data_name == 'diginetica'):
            train_data, test_data = self.change_time_feature(train_data, test_data)

        train_data = self.add_time_intervals(train_data)
        test_data = self.add_time_intervals(test_data)
        
        train_data = self.add_h_m_s(train_data, "a")
        train_data = self.add_h_m_s(train_data, "b")

        test_data = self.add_h_m_s(test_data, "a")
        test_data = self.add_h_m_s(test_data, "b")

        if(len(train_data) != len_train and len(test_data) != len_test):
            raise ValueError('The size of the train and test sets have been changed during the preprocessing')
        return train_data, test_data
        
    def add_h_m_s(self, data, delta_type):
        data["h_" + delta_type] = 24
        data["m_" + delta_type] = 60
        data["s_" + delta_type] = 60
        
        delta_t_values = data["delta_t_" + delta_type].values
        delta_t_values[(delta_t_values > 86399)] = 86399
        
        minute, sec = np.divmod(delta_t_values, 60)
        hour , minute = np.divmod(minute, 60)
        day , hour = np.divmod(hour, 24)
    
        non_zero_mask = (delta_t_values >= 0)
        data.loc[(non_zero_mask), ("s_" + delta_type)] = sec[non_zero_mask]
        data.loc[(non_zero_mask), ("m_" + delta_type)] = minute[non_zero_mask]
        data.loc[(non_zero_mask), ("h_" + delta_type)] = hour[non_zero_mask]        
    
        return data
    
    def add_time_intervals(self, data):
        data.sort_values(['session_id', self.time_key], inplace=True)
        data = data.reset_index(drop=True) 
        
        times = data[self.time_key].values.copy()
        times = np.roll(times, -1)
        intervals = np.ceil(times - data[self.time_key].values)
        
        session_keys = data[self.session_key].values.copy()
        session_keys = np.roll(session_keys, -1)
        session_diff = data[self.session_key].values - session_keys
        no_interval_mask = (session_diff != 0)
        
        intervals[no_interval_mask] = 0.0
        data['delta_t_a'] = deepcopy(intervals)
        
        times = data[self.time_key].values.copy()
        times = np.roll(times, 1)
        intervals = np.ceil(data[self.time_key].values - times)
        
        session_keys = data[self.session_key].values.copy()
        session_keys = np.roll(session_keys, 1)
        session_diff = data[self.session_key].values - session_keys
        no_interval_mask = (session_diff != 0)
        
        intervals[no_interval_mask] = 0.0
        data['delta_t_b'] = deepcopy(intervals)

        return data
    
    def save(self, data, data_dir, name):             
        data_dir = os.path.join(data_dir, self.data_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_path = os.path.join(data_dir, name)
        data = self.sort_events(data)
        data.to_pickle(data_path)
            
    def remove_non_common_items(self, data, test_data):
        mask = np.in1d(test_data[self.item_key], data[self.item_key])
        test_data = test_data[mask]
        return test_data
    
    def load_data(self, path):
        if(self.data_name == "yoochoose"):
            path = os.path.join(path, "yoochoose-clicks.dat")
            data = pd.read_csv(path, sep=',', header=None,
                    usecols=[0, 1, 2, 3], dtype={0:np.int32, 1:str, 2:np.int64, 3:str})
            data.columns = [self.session_key, self.time_key, self.item_key, self.cat_key]
        else :
            path = os.path.join(path, "train-item-views.csv")
            data = pd.read_csv(path, sep=';', header=0,
                   usecols=[0, 2, 3, 4], dtype={0:np.int32, 2:np.int64, 3:np.int64, 4:str})
            data.columns = [self.session_key, self.item_key, 'timeframe', self.time_key]
        return data

    def resolve_time_stamps(self, data):
        data.sort_values([self.session_key, 'timeframe'], inplace=True)
        data = data.reset_index(drop=True)
        first_time = -1
        current_session = -1
        for row_index , data_row in data.iterrows():
            if(data_row[self.session_key] != current_session):            
                first_time = data_row[self.time_key]
                current_session = data_row[self.session_key]
                continue
            else:
                data.at[row_index, self.time_key] = first_time + data_row['timeframe']                
        return data            
        
    def add_time_feature(self, data):
        if(self.data_name == "yoochoose"):            
            data['time'] = data[self.time_key].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())    
        else :
            data['time'] = data[self.time_key].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())
            data['timeframe'] = data['timeframe'].apply(lambda x: (x / 1000))
            
        del(data[self.time_key])
        self.time_key = 'time'
        return data
    
    def change_time_feature(self, train_data, test_data):
        train_data = self.resolve_time_stamps(train_data)
        test_data = self.resolve_time_stamps(test_data)
        
        train_data = self.sort_events(train_data)
        test_data = self.sort_events(test_data)

        return train_data, test_data
    
    def remove_short_sessions(self, data, min_len):        
        sessionLen = data.groupby(self.session_key).size()  
        mask = np.in1d(data[self.session_key], sessionLen[sessionLen >= min_len].index)
        data = data[mask]
        return data
        
    def remove_rare_items(self, data, min_count):
        item_supports = data.groupby(self.item_key).size()
        mask = np.in1d(data[self.item_key], item_supports[item_supports >= min_count].index)
        data = data[mask]
        return data
        
    def sort_events(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        return data

    def get_avg_len(self, data):
        sizes = data.groupby(self.session_key).size().values
        n_sequence = np.sum((sizes - 1))
        sum_session_sizes = np.sum((sizes * (sizes + 1)) / 2)    
        session_mean = (sum_session_sizes / n_sequence)
        return session_mean
    
    def get_n_sequence(self, data):
        n_sequence = (len(data) - data[self.session_key].nunique())
        return n_sequence
        
    def get_test_data(self, data):
        t_max = data[self.time_key].max()
        if(self.data_name == "yoochoose"):
            session_times = data.groupby(self.session_key)[self.time_key].max()
            splitdate = t_max - 86400 * 1 
        else :
            session_times = data.groupby(self.session_key)[self.time_key].max()
            splitdate = t_max - 86400 * 7
            
        train_ids = session_times[session_times < splitdate].index
        test_ids = session_times[session_times > splitdate].index
        
        before_last_day = data[np.in1d(data[self.session_key], train_ids)]
        last_day = data[np.in1d(data[self.session_key], test_ids)]
    
        return before_last_day, last_day

    def sort_sessions(self, data):
        session_times = data.groupby(self.session_key)[self.time_key].min()
        session_times = pd.DataFrame({self.session_key:session_times.index,
                                      'session_time':session_times.values})
        data = pd.merge(data, session_times, on=self.session_key, how='inner')
        
        data.sort_values(['session_time', self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        del(data['session_time'])
        return data
    
    def fraction(self, data):
        data = self.sort_sessions(data)
        n_sequence = self.get_n_sequence(data)
        n_sequence_1_4 = int(n_sequence / 4)
        n_sequence_1_64 = int(n_sequence / 64)
        session_ids = data[self.session_key].values
        cur_session_id = -1
        n_seq = 0
        for i in range(len(data) - 1, -1, -1):
            n_seq += 1
            if cur_session_id != session_ids[i]:
                n_seq -= 1
            cur_session_id = session_ids[i]
            if (n_seq == n_sequence_1_4):
                split_index_1_4 = i
                break
            if(n_seq == n_sequence_1_64):
                split_index_1_64 = i
                n_sequence_1_64 = -1
                
        train_1_4 = deepcopy(data.iloc[split_index_1_4:])
        train_1_64 = deepcopy(data.iloc[split_index_1_64:])
        return train_1_4, train_1_64
    
    
    def get_n_clicks(self, data):
        n_clickes = len(data)
        return n_clickes
    
    def get_n_sessions(self, data):
        n_sessions = data[self.session_key].nunique()
        return  n_sessions
    
    def get_n_items(self, data):
        n_items = data[self.item_key].nunique()
        return n_items
    
    def print_stat(self, data, data_name):
        avg_len = self.get_avg_len(data)
        n_clicks = self.get_n_clicks(data)
        n_items = self.get_n_items(data)
        n_sessions = self.get_n_sequence(data)
        
        print("#########################################")
        print("Dataset: {} \n# of all the clicks: {} \n# of sessions: {} \n"
            "# of items: {} \nAverage length: {:.2f} \n".format(
                (data_name),
                n_clicks,
                n_sessions,
                n_items,
                avg_len))
