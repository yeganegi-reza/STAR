import os
import torch
import random
import argparse
import numpy as np

from model import STAR
from tools import Trainer
from tools import DataSet

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='diginetica', 
                    help='dataset name: diginetica/yoochoose')
parser.add_argument('--fraction', default=4, help='1/4/64')
parser.add_argument('--validation', type=bool, default=False, 
                    help='use validation or not')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--do_pretrain', type=bool, default=True, help='item embedding dim')
parser.add_argument('--iter', type=int, default=100, 
                    help='number of epochs for glove')
parser.add_argument('--theta_coef', type=int, default=2.0, 
                    help='coefficient to control the time interval between actions')

parser.add_argument('--embedding_dim', type=int, default=180, help='item embedding dim')
parser.add_argument('--hidden_size', type=int, default=180, help='hidden state size')

parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--epoch', type=int, default=12, help='the number of epochs to train for')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
 
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=6, 
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--w_dc', type=int, default=1e-6, help='weight decay rate')
parser.add_argument('--top_k', type=int, default=20, help='k in metrics')
parser.add_argument('--seed_value', type=int, default=8, help='seed')

args = parser.parse_args()
   
if __name__ == "__main__":
    torch.manual_seed(args.seed_value)
    torch.cuda.manual_seed(args.seed_value)
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)  
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                     
    data_dir = os.path.join("Data", "cleaned" , args.data_name)
    result_dir = os.path.join('Results', args.data_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
                   
    if args.data_name == 'diginetica':
        n_items = 43097
    elif args.data_name == 'yoochoose':
        n_items = 37483 
        
    '''create dataset'''
    train_dataset, test_dataset = DataSet.create_sets(data_dir, args, n_items)
    
    '''create STAR Framework'''
    star_framework = STAR.STARFramework(device, n_items, args)
    star_framework.set_optimizer(args)
       
    trainer = Trainer.Trainer()
    trainer.train(star_framework, train_dataset, test_dataset, args, result_dir)
      
    checkPoints = {'model': star_framework}
    save_path = os.path.join(result_dir, "Star_model")
    torch.save(checkPoints, save_path)

