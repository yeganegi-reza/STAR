import os
import pandas as pd
import numpy as np
from math import ceil
import random
from copy import deepcopy


class Dataset(object):
    def __init__(self, data=None, data_path="", args={},
                 n_items=0, train_set=True):
 
        self.batch_size = args.batch_size
        self.train_set = train_set
        self.data_name = args.data_name
        self.n_items = n_items
        self.padd_idx = n_items
        self.set_keys()
        
        if(data is None):
            self.data = self.load_data(data_path)
        else :
            self.data = data
        
        self.sort_data();
        self.max_session_len = self.get_max_sequence_len(self.data)
        
    def load_data(self, data_path):
        print("Loading data from {}".format(data_path))
        data = pd.read_pickle(data_path)
        return data
        
    def set_keys(self):
        self.session_key = "session_id"
        self.item_idx = 'item_idx'
        self.time_key = "time"

    def sort_data(self):
        self.data.sort_values([self.session_key, self.time_key], inplace=True)
        self.data = self.data.reset_index(drop=True)
    
    def create_offset_sessions(self, data):
        offset_sessions = (data[self.session_key].values 
                          - np.roll(data[self.session_key].values, 1))
        offset_sessions = np.nonzero(offset_sessions)[0]
        return offset_sessions

    def get_max_sequence_len(self, data):
        return max(data.groupby(self.session_key).size())

    def sort_session_by_len(self, data):
        size_key = "session_size"
        session_len = data.groupby(self.session_key).size().sort_values()
        session_len = session_len.to_frame(name=size_key)
        data = pd.merge(data, session_len, on=self.session_key, how='inner')
        data.sort_values([size_key, self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        return data
        
    # this method get index of the first session in each group of sessions with same length
    def get_len_offsets(self, data):
        data = self.sort_session_by_len(data)
        size_key = "session_size"    
        lengths = data[size_key].values
        lengths = np.roll(lengths, 1)
        lengths = lengths - data[size_key].values
        len_change_idx = np.sort(np.nonzero(lengths != 0)[0])
        offset_lenghts = data[size_key].values[len_change_idx]
        del[data[size_key]]
        return data, np.vstack((offset_lenghts, len_change_idx)).T

    def __iter__(self):
        """ Returns the iterator for producing mini-batches"""
        for (item_sequence, target, h_a, m_a, s_a,
             h_b, m_b, s_b) in self.get_mini_batch():
            yield (item_sequence, target, h_a, m_a, s_a,
                   h_b, m_b, s_b)
    
   
    def get_mini_batch(self):  
        lengths_list = np.random.permutation(np.arange(2, self.max_session_len + 1))
        
        data, start_indices = self.get_len_offsets(self.data)
        offset_sessions = self.create_offset_sessions(data)
        max_session_len = self.get_max_sequence_len(data)
            
        item_idx = data[self.item_idx].values;
        h_a_idx = data['h_a'].values
        m_a_idx = data['m_a'].values
        s_a_idx = data['s_a'].values
        
        h_b_idx = data['h_b'].values
        m_b_idx = data['m_b'].values
        s_b_idx = data['s_b'].values
        
        # place same length sessions in a batch
        for length in lengths_list:
            if(length > max_session_len):
                continue
            if(length < 2):
                raise "invalid session size"

            start_index = np.argmax(start_indices[:, 0] >= length)
            start_index = start_indices[start_index, 1]
            session_starts = offset_sessions[offset_sessions >= start_index]            
            np.random.shuffle(session_starts)
            batch_size = self.batch_size
            
            # Order of sessions
            n_sessions = len(session_starts) 
            session_idx_arr = np.arange(n_sessions);
        
            if(batch_size > n_sessions):
                batch_size = n_sessions

            iters = np.arange(batch_size)
            stop = False

            h_b = np.zeros([batch_size, length - 1]) 
            m_b = np.zeros([batch_size, length - 1]) 
            s_b = np.zeros([batch_size, length - 1]) 

            h_a = np.zeros([batch_size, length - 1])
            m_a = np.zeros([batch_size, length - 1])
            s_a = np.zeros([batch_size, length - 1])
            
            item_sequence = np.zeros([batch_size, length - 1])
        
            while not stop:
                
                batch_size = len(iters)
                start = session_starts[session_idx_arr[iters]]   
                end = start + length - 1
                
                item_sequence[:] = 0
                
                for k in range(batch_size):
                    item_sequence[k, :] = item_idx[start[k]:end[k]]

                    h_a[k, :] = h_a_idx[start[k]:(end[k])]
                    m_a[k, :] = m_a_idx[start[k]:(end[k])]
                    s_a[k, :] = s_a_idx[start[k]:(end[k])]
                    
                    h_a[k, length - 2] = 0
                    m_a[k, length - 2] = 0
                    s_a[k, length - 2] = 0

                    h_b[k, :] = h_b_idx[(start[k]):end[k]]
                    m_b[k, :] = m_b_idx[(start[k]):end[k]]
                    s_b[k, :] = s_b_idx[(start[k]):end[k]]
                    h_b[k, 0] = 0
                    m_b[k, 0] = 0
                    s_b[k, 0] = 0
                    
                    
                out_idx = item_idx[end]
                
                item_sequence_o = item_sequence[0:batch_size]                    
                target = out_idx[0:batch_size]
                h_a_o = h_a[0:batch_size]
                m_a_o = m_a[0:batch_size]
                s_a_o = s_a[0:batch_size]

                h_b_o = h_b[0:batch_size]
                m_b_o = m_b[0:batch_size]
                s_b_o = s_b[0:batch_size]
                
                yield (item_sequence_o, target, h_a_o, m_a_o, s_a_o,
                        h_b_o, m_b_o, s_b_o)
                
                iters = iters + batch_size 
                iters = iters[iters < n_sessions]
                
                if (len(iters) == 0):
                    stop = True;
                    break;
       
                sessions = session_idx_arr[iters]
                start = session_starts[sessions]
                end = start + length - 1

    def get_validation(self, partition):
        if(self.train_set):
            
            self.data = self.sort_sessions(self.data)
            n_sequence = (len(self.data) - self.data[self.session_key].nunique())
            n_sequence = int(n_sequence / (partition * 100))
            session_ids = self.data[self.session_key].values
            cur_session_id = -1
            n_seq = 0
            for i in range(len(self.data) - 1, -1, -1):
                n_seq += 1
                if cur_session_id != session_ids[i]:
                    n_seq -= 1
                cur_session_id = session_ids[i]
                if (n_seq == n_sequence):
                    split_index = i
                    break
                                        
            valid_data = deepcopy(self.data.iloc[split_index:])
            self.data = deepcopy(self.data.iloc[:split_index])
            
            print(len(self.data))
            return valid_data
        else :
            raise
    
    def sort_sessions(self, data):
        session_times = data.groupby(self.session_key)[self.time_key].min()
        session_times = pd.DataFrame({self.session_key:session_times.index,
                                      'session_time':session_times.values})
        data = pd.merge(data, session_times, on=self.session_key, how='inner')
        
        data.sort_values(['session_time', self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        del(data['session_time'])
        return data
    
def create_sets(data_dir, args, n_items):

    if(args.data_name == "yoochoose"):
        train_path = os.path.join(data_dir, "train_1_" + str(args.fraction) + ".pkl")
    else:
        train_path = os.path.join(data_dir, "train_1_1.pkl")
        
    train_dataset = Dataset(data=None, data_path=train_path, args=args,
                            n_items=n_items)
        
    if(not args.validation):
        test_path = os.path.join(data_dir, "test.pkl")
        test_dataset = Dataset(data=None, data_path=test_path, args=args,
                               n_items=n_items, train_set=False)

    else :
        test_dataset = Dataset(data=train_dataset.get_validation(args.valid_portion),
                               data_path="", args=args,
                               n_items=n_items, train_set=False)
     
    return train_dataset, test_dataset