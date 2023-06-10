import torch
from torch import nn

from tools import Metrics
from tools import Optimizer

from model.Modules import Module_1_6
from model.Modules import Module_2
from model.Modules import Module_3
from model.Modules import Module_4
from model.Modules import Module_5


class STAR(nn.Module):
    def __init__(self, device, n_items, args):
        super(STAR, self).__init__()
        
        self.device = device
        self.module_1_6 = Module_1_6(self.device, n_items, args.embedding_dim, args.dropout)
        self.module_2 = Module_2(self.device, args.embedding_dim, args.hidden_size,
                                 args.dropout)
        
        self.module_3 = Module_3(self.device, args.embedding_dim, args.dropout)
        self.module_4 = Module_4(self.device, args.dropout)
        self.module_5 = Module_5(self.device, args.embedding_dim, args.dropout)

        self = self.to(self.device)

    def forward(self, item_sequence,
                h_a_o, m_a_o, s_a_o,
                h_b_o, m_b_o, s_b_o,):

        embeddings = self.module_1_6(item_sequence,
                                     h_a_o, m_a_o, s_a_o,
                                     h_b_o, m_b_o, s_b_o)
        
        hidden_states = self.module_2(embeddings[0])
        b_embedding, a_embedding = self.module_3(embeddings[1], embeddings[2], embeddings[3],
                                                 embeddings[4], embeddings[5], embeddings[6])
        
        weighted_features = self.module_4(hidden_states, b_embedding, a_embedding)
        final_hidden = self.module_5(weighted_features[0], weighted_features[1],
                                     weighted_features[2], weighted_features[3],
                                     weighted_features[4], weighted_features[5])
        Y_tilda = self.module_1_6.get_scores(final_hidden)
        return Y_tilda
    
    def set_inital_weights(self, initial_weights):
        self.module_1_6.set_initial_embedds(initial_weights)
        
class STARFramework(nn.Module):
    def __init__(self, device, n_items, args):
        super(STARFramework, self).__init__()
        self.star = STAR(device, n_items, args)
        self.n_items = n_items
        self.device = device
        self.set_loss()
    
    def set_loss(self):
        self.loss_fun = torch.nn.CrossEntropyLoss()
      
    def set_optimizer(self, args):
        self.optimizer = Optimizer.Optimizer(filter(
            lambda p: p.requires_grad, self.star.parameters()), args) 
          
    def set_inital_weights(self, FE_initial_weights):
        self.star.set_inital_weights(FE_initial_weights)
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()
    
    def step_scheduler(self):
        self.optimizer.step_scheduler()

    def forward(self, item_sequence,
                h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        
        Y_hat = self.star(item_sequence,
                          h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)
        return Y_hat
    
    def calc_loss(self, Y_hat, target, batch_size):
        current_batch_size = target.shape[0]        
        loss = self.loss_fun(Y_hat, target)

        # All batches are are not same size
        loss = (float(current_batch_size) / float(batch_size)) * loss                    
        return loss
    
    def fit(self, item_sequence, target,
            h_a_o, m_a_o, s_a_o,
            h_b_o, m_b_o, s_b_o,
            batch_size):
        
        target = torch.LongTensor(target).to(self.device)                
        Y_hat = self.forward(item_sequence,
                               h_a_o, m_a_o, s_a_o,
                               h_b_o, m_b_o, s_b_o)
        
        loss = self.calc_loss(Y_hat, target, batch_size)
        self.backward(loss)
        return loss
        
    def log_train(self, epoch, train_loss, valid_loss, recall, mrr, epoch_time):
        print("Epoch: {}, train loss: {:.4f}, validloss: {:.4f}, recall: {:.4f}, "
            "mrr: {:.4f}, time: {:.2f}".format(
                epoch, train_loss, valid_loss, recall, mrr, epoch_time))

    def log_test(self, loss, recall, mrr):
        print("Test: loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}".format(loss, recall, mrr))
    
    def test(self, item_sequence, target,
              h_a_o, m_a_o, s_a_o,
              h_b_o, m_b_o, s_b_o,
              batch_size, top_k): 
        
        with torch.no_grad():
            target = torch.LongTensor(target).to(self.device)                
            Y_hat = self.forward(item_sequence,
                                   h_a_o, m_a_o, s_a_o,
                                   h_b_o, m_b_o, s_b_o)
            
            loss = self.calc_loss(Y_hat, target, batch_size)
            recall, mrr = Metrics.calc(Y_hat, target, k=top_k)
            return loss, recall, mrr
