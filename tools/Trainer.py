import os
import time
import numpy as np
from model import Embedding

class Trainer():
    
    def Pretrain(self, dataset, args, framework, result_dir):
        pre_train_model = Embedding.GloveModel(args.embedding_dim, args.iter)
        fraction = args.fraction
        if(args.do_pretrain):
            '''train initial embeddings'''
            pre_train_model.train(dataset, args.theta_coef)
            save_path = os.path.join(result_dir, ("initial_embeddings_" + str(fraction)))
            pre_train_model.save_weights(save_path)        
        else:
            load_path = os.path.join(result_dir, ("initial_embeddings_" + str(fraction)))
            if not os.path.exists(result_dir):
                raise Exception("Initial Embeddings Not Exist")
            pre_train_model.load_weights(load_path)
        initial_embedds = pre_train_model.get_weights()                    
        framework.set_inital_weights(initial_embedds)
        del pre_train_model
    
    def train(self, framework, train_data, valid_data, args, result_dir):
        self.Pretrain(train_data, args, framework, result_dir)
        
        epochs = args.epoch
        batch_size = args.batch_size
        
        print("Start Star Training")
        for epoch in range(epochs):
            print('Start Epoch #', epoch)
            framework.train()
            st = time.time()
            loss_list = []
            for (item_sequence, target,
                 h_a, m_a, s_a,
                 h_b, m_b, s_b) in train_data:
                
                loss = framework.fit(item_sequence, target,
                                      h_a, m_a, s_a,
                                      h_b, m_b, s_b, batch_size)    
                loss_list.append(loss)
            epoch_time = time.time() - st
            loss_list = [loss.cpu().item() for loss in loss_list]
            train_loss = np.mean(loss_list)
            valid_loss , recall, mrr = self.eval(framework, valid_data, args)
            framework.log_train(epoch, train_loss, valid_loss, recall, mrr, epoch_time)
            framework.step_scheduler()
            
    def eval(self, framework, valid_data, args):
        total_samples = 0
        batch_size = valid_data.batch_size
        loss_list_valid = []
        recall_list = []
        mrr_list = []
        framework.eval()
        
        for (item_sequence, target,
             h_a_o, m_a_o, s_a_o,
             h_b_o, m_b_o, s_b_o) in valid_data:
            
            total_samples += target.shape[0]
            
            loss, recall, mrr = framework.test(item_sequence, target,
                                               h_a_o, m_a_o, s_a_o,
                                               h_b_o, m_b_o, s_b_o,
                                               batch_size, args.top_k)
            loss_list_valid.append(loss)
            recall_list.append(recall)
            mrr_list.append(mrr)
            
        loss_list_valid = [loss.cpu().item() for loss in loss_list_valid]
        valid_loss = np.mean(loss_list_valid)
        mean_recall = np.sum(recall_list) / float(total_samples)
        mean_mrr = np.sum(mrr_list) / float(total_samples)
        return valid_loss, mean_recall, mean_mrr
