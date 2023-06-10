import os
import argparse
from tools import Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='diginetica', 
                    help='dataset name: diginetica/yoochoose')
parser.add_argument('--min_len', type=int, default=2, 
                    help='minimum number of items in sessions')
parser.add_argument('--min_count', type=int, default=5, 
                    help='minimum number items in datasets')


args = parser.parse_args()
if __name__ == '__main__':
    data_path = 'Data'
    raw_data_dir = os.path.join(data_path, "raw", args.data_name)
    cleaned_data_dir = os.path.join(data_path, "cleaned")
            
    pre_pro = Preprocessor.Preprocessor()
    pre_pro.process(args, raw_data_dir, cleaned_data_dir)