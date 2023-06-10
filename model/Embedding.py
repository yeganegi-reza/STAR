import pickle
import numpy as np

from copy import deepcopy
from glove import Corpus, Glove
from math import floor
class GloveModel(object):
    def __init__(self, embedding_dim, epoch):
        self.dim = embedding_dim
        self.epoch = epoch

    def train(self, dataset, theta_coef):        
        sentences, max_sent_size = self.create_sentences(dataset, theta_coef)
        print("Start GloVe Training")
        corpus = Corpus() 
        corpus.fit(sentences, window=max_sent_size)
        
        glove = Glove(no_components=self.dim, learning_rate=0.1)
        glove.fit(corpus.matrix, epochs=self.epoch, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)

        print("End GloVe Training")
        self.embedds = self.set_embedds(glove)
    
    def get_mean_interval(self, data):
        time_intervals = data['delta_t_a'].values.copy()        
        session_keys = data['session_id'].values.copy()
        session_keys = np.roll(session_keys, -1)
        session_diff = data['session_id'].values - session_keys
        interval_mask = (session_diff == 0)
        mean_interval = time_intervals[interval_mask].mean()        
        return mean_interval
        
        
    def create_sentences(self, dataset, theta_coef):
        mean_interval = self.get_mean_interval(dataset.data)
        theta = floor(theta_coef * mean_interval)
        data = deepcopy(dataset.data)
        groups = data.groupby(dataset.session_key)
        sentences = groups.apply(self.get_sentences, theta)
        sentences = sentences.values;
        sentences_list = [sent for sent in sentences]
        sentences = []
        max_sent_size = 0
        for sent in sentences_list:
            for s in sent:
                if(len(s) > max_sent_size):
                    max_sent_size = len(s)
                sentences.append(s)
        return sentences, max_sent_size
    
    def get_sentences(self, group, theta):
        items = group['item_idx'].values
        times = group['time'].values
        sentences = []
        sent = []
        last_action_time = times[0]
        sequence_size = len(items)
        n_remain = sequence_size
        for i in range(sequence_size):
            if((abs(times[i] - last_action_time) < theta)):
                sent.append(str(items[i]))
                last_action_time = times[i]
            else :
                if(len(sent)):
                    sentences.append(sent)
                sent = []
                sent.append(str(items[i]))
                last_action_time = times[i]
            n_remain -= 1
        if(len(sent)):
            sentences.append(sent)
        return sentences
    
    def set_embedds(self, model):
        embeddings = {}
        for key in model.dictionary:
            embeddings[key] = model.word_vectors[model.dictionary[key]];
        return embeddings
    
    def get_weights(self):
        return deepcopy(self.embedds);
 
    def save_weights(self, save_path):
        pickle.dump(self.embedds, open(save_path, 'wb'))
    
    def load_weights(self, load_path):
        self.embedds = pickle.load(open(load_path, "rb"))
