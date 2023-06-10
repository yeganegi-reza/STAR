import math
import torch
from torch import nn
from model import Component as Com

#Feature Extraction, Preference Detection
class Module_1_6(Com.Component):
    def __init__(self, device, n_items, embedding_dim, dropout):
        
        super().__init__(device)
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.designModelArch(dropout)

    def designModelArch(self, dropout):
        n_embeddings = self.n_items
        self.item_embedding = nn.Embedding((n_embeddings + 1), self.embedding_dim,
                                           padding_idx=n_embeddings)
                
        self.hour_embedding_a = nn.Embedding(25, self.embedding_dim, padding_idx=24)
        self.minute_embedding_a = nn.Embedding(61, self.embedding_dim, padding_idx=60)
        self.second_embedding_a = nn.Embedding(61, self.embedding_dim, padding_idx=60)

        self.hour_embedding_b = nn.Embedding(25, self.embedding_dim, padding_idx=24)
        self.minute_embedding_b = nn.Embedding(61, self.embedding_dim, padding_idx=60)
        self.second_embedding_b = nn.Embedding(61, self.embedding_dim, padding_idx=60)
                
        self.bias_last = torch.nn.Parameter(torch.Tensor(self.n_items))                
        self.dropout = nn.Dropout(dropout)
                
    def set_initial_embedds(self, initial_embedds):
        for item_id, embedding in initial_embedds.items():
            self.item_embedding.weight.data[int(item_id)].copy_(
                torch.from_numpy(embedding).float())        

        norms = torch.norm(self.item_embedding.weight[:self.n_items], p=2, dim=1).data
        self.item_embedding.weight.data[:self.n_items] = self.item_embedding.weight.data[:self.n_items].div(norms.view(-1, 1).expand_as(self.item_embedding.weight[:self.n_items]))
        
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.item_embedding.weight[:self.n_items])
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_last, -bound, bound) 
    
    def get_scores(self, final_hidden):        
        scores = torch.matmul(final_hidden,
                              self.item_embedding.weight[:(self.n_items)].transpose(1, 0))
        scores = scores + self.bias_last
        return scores
        
    def forward(self, session,
                h_a_o, m_a_o, s_a_o,
                h_b_o, m_b_o, s_b_o):

        h_b_o = torch.LongTensor(h_b_o).to(self.device)
        m_b_o = torch.LongTensor(m_b_o).to(self.device)
        s_b_o = torch.LongTensor(s_b_o).to(self.device)
  
        h_a_o = torch.LongTensor(h_a_o).to(self.device)
        m_a_o = torch.LongTensor(m_a_o).to(self.device)
        s_a_o = torch.LongTensor(s_a_o).to(self.device)
  
        h_b_embedding = self.hour_embedding_b(h_b_o)
        m_b_embedding = self.minute_embedding_b(m_b_o)
        s_b_embedding = self.second_embedding_b(s_b_o)
          
        h_a_embedding = self.hour_embedding_a(h_a_o)
        m_a_embedding = self.minute_embedding_a(m_a_o)
        s_a_embedding = self.second_embedding_a(s_a_o)
        
        session = torch.LongTensor(session).to(self.device)
        session_embedding = self.item_embedding(session)
        session_embedding = self.dropout(session_embedding)
        all_embeddings = [session_embedding, h_b_embedding, m_b_embedding, s_b_embedding,
                 h_a_embedding, m_a_embedding, s_a_embedding]
        
        return all_embeddings

#Session Encoding
class Module_2(Com.Component):
    def __init__(self, device, embedding_dim, hidden_size, dropout=0):
        super().__init__(device)
        self.designModelArch(embedding_dim, hidden_size, dropout)

    def designModelArch(self, embedding_dim , hidden_size, dropout):
        gru_input_size = embedding_dim
        self.gru = nn.GRU(gru_input_size, hidden_size, 1, bias=True,
                          batch_first=True, bidirectional=True) 
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
             
    def forward(self, E_session):            
        gru_out , h_t = self.gru(E_session)
        gru_out = gru_out[:, :, :self.hidden_size] + gru_out[:, :, self.hidden_size:]
        gru_out = (gru_out / 2)
        gru_out = self.dropout(gru_out)
        return gru_out

#Attention Weight Calculation
class Module_3(Com.Component):
    def __init__(self, device, embedding_dim, dropout):
        super().__init__(device)
        self.designModelArch(embedding_dim, dropout)

    def designModelArch(self, embedding_dim, dropout):        
        self.linear_b = nn.Linear((3 * embedding_dim), embedding_dim)
        self.linear_a = nn.Linear((3 * embedding_dim), embedding_dim)                
        self.weight_fun = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
                        
    def forward(self, h_b_embedding, m_b_embedding, s_b_embedding,
                h_a_embedding, m_a_embedding, s_a_embedding):
       
        before_embedding = torch.cat((h_b_embedding, m_b_embedding, s_b_embedding), 2)
        before_embedding = self.dropout(before_embedding)
        before_embedding = self.linear_b(before_embedding)
        before_embedding = self.dropout(before_embedding)
        before_embedding = self.weight_fun(before_embedding)
          
        after_embedding = torch.cat((h_a_embedding, m_a_embedding, s_a_embedding), 2)
        after_embedding = self.dropout(after_embedding)
        after_embedding = self.linear_a(after_embedding)
        after_embedding = self.dropout(after_embedding)
        after_embedding = self.weight_fun(after_embedding)
                       
        return before_embedding, after_embedding

#Time Attention
class Module_4(Com.Component):
    def __init__(self, device, dropout):
        super().__init__(device)
        self.designModelArch(dropout)

    def designModelArch(self, dropout):
        self.dropout = nn.Dropout(dropout)
             
    def forward(self, gru_out, E_Before, E_After):
        gru_out_A_final = E_After.mul(gru_out)
        h_t_a_m = gru_out_A_final[:, -1, :]
        h_t_a_1 = gru_out_A_final[:, 0, :]
        
        gru_out_B_final = E_Before.mul(gru_out)
        h_t_b_m = gru_out_B_final[:, -1, :]
        h_t_b_1 = gru_out_B_final[:, 0, :]
        
        alpha_A = torch.bmm(gru_out_A_final, h_t_a_m.unsqueeze(2)).squeeze(2) / 11
        alpha_A = torch.softmax(alpha_A, 1)
        AP_A = torch.bmm(gru_out_A_final.transpose(1, 2), alpha_A.unsqueeze(2)).squeeze(2)                
  
        alpha_A_2 = torch.bmm(gru_out_A_final, h_t_a_1.unsqueeze(2)).squeeze(2) / 11
        alpha_A_2 = torch.softmax(alpha_A_2, 1)
        AP_A_2 = torch.bmm(gru_out_A_final.transpose(1, 2), alpha_A_2.unsqueeze(2)).squeeze(2)                
   
        alpha_B = torch.bmm(gru_out_B_final, h_t_b_m.unsqueeze(2)).squeeze(2) / 11
        alpha_B = torch.softmax(alpha_B, 1)
        AP_B = torch.bmm(gru_out_B_final.transpose(1, 2), alpha_B.unsqueeze(2)).squeeze(2)                
  
        alpha_B_2 = torch.bmm(gru_out_B_final, h_t_b_1.unsqueeze(2)).squeeze(2) / 11
        alpha_B_2 = torch.softmax(alpha_B_2, 1)
        AP_B_2 = torch.bmm(gru_out_B_final.transpose(1, 2), alpha_B_2.unsqueeze(2)).squeeze(2)                
        
        weighted_features = [h_t_a_m, h_t_b_m, AP_A, AP_B, AP_A_2, AP_B_2]
        return weighted_features

#Self Attention
class Module_5(Com.Component):
    def __init__(self, device, embedding_dim, dropout):
        
        super().__init__(device)
        self.designModelArch(dropout, embedding_dim)

    def designModelArch(self, dropout, embedding_dim):        
        self.linear = nn.Linear((embedding_dim * 6), embedding_dim, bias=True)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h_t_a, h_t_b, AP_A, AP_B, AP_A_2, AP_B_2):
        concate_vec = torch.cat((h_t_a, h_t_b, AP_A, AP_B, AP_A_2, AP_B_2), 1)
        final_hidden = self.linear(concate_vec)
        final_hidden = self.dropout(final_hidden)
        return final_hidden