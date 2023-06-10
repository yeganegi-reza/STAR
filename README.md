# STAR

- PyTorch implementation of the algorithm of [A Session-Based Time-Aware Recommender System](https://arxiv.org/abs/2211.06394). 

## Requirements

- Python 3.9
- PyTorch 1.11.0
- Pandas 1.4.3
- Numpy 1.21.5
-  [Glove](https://github.com/maciejkula/glove-python)

## Usage

### Dataset

- Here are two datasets we used in the paper. After downloading the datasets, put them in the folder `Data/Raw/`:
- [YOOCHOOSE](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
- [DIGINETICA](https://competitions.codalab.org/competitions/11161)

### Data Preprocessing 

 Run `preprocess.py` for data preprocessing and cleaning.

- After preprocessing, `train and test` sets are obtained and stored in`(Data/Cleaned)`

### Train And Test 

Run `main.py` for training and evaluating the model.


**Model Parameters**
   - The following list of parameters 

   - ```data_name```  dataset name: diginetica/yoochoose (Default = diginetica) <br>
     ```fraction``` The fraction of data for yoochoose dataset (4,64) (Default = 4) <br>
     ```validation``` Using validation or not (Default = False) <br>
     ```valid_portion``` split the portion of training set as validation set (Default = 0.1) <br>
     ```do_pretrain``` Whether do pretrain by GLOVE or not (Default = True) <br>
     ```iter``` The Number of epochs for Glove Method (Default = 100) <br>
     ```theta_coef``` Coefficient to control the time interval between actions (Default = 2.0) <br>
	```embedding_dim``` The dimension of embeddings (Default = 180) <br>
     ```hidden_size``` The dimension of hidden states (Default = 180)<br>
     ```batch_size``` Input batch size (Default: 512) <br>
     ```epoch``` The number of epochs (Default = 12) <br>
     ```dropout``` Dropout rate <br>
     ```lr``` Learning rate (Default = 1e-3).<br>
     ```lr_dc``` learning rate decay rate (Default = 0.1) <br>
     ```lr_dc_step``` The number of epochs after which the learning rate decay (Default = 6) <br>
     ```w_dc``` Weight decay rate (Default = 1e-6)<br>
     ```top_k```  Value of K used during Recall@K and MRR@K Evaluation (Default = 20)<br>
     ```seed_value``` Seed value (Default = 8.0) <br>

### Citation
Please cite our paper if you use the code:

```
@inproceedings{yeganegi2022star,
    title={STAR: A Session-Based Time-Aware Recommender System},
    author={Reza Yeganegi and Saman Haratizadeh},
    year={2022},
    eprint={2211.06394},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```
