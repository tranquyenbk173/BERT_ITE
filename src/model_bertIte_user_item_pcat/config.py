import csv
import datetime
import scipy.sparse as sp
import numpy as np
import csv
import datetime
import scipy.sparse as sp
import numpy as np
import progressbar
import random
import torch

class Config:

    def __init__(self, root_path, params):
        
        self.attn_pdrop = params['attn_pdrop']
        self.resid_pdrop = params['resid_pdrop']
        self.n_head = params['n_head']
        self.n_layer = params['n_layer']
        self.interval = params['interval']
        self.eta = params['eta']
        self.reg_lambda = params['reg_lambda']
        self.num_neg = params['num_neg']
        self.max_len = params['max_len']
        self.eval_top_k = params['eval_top_k']
        self.batch_size = params['batch_size']
        self.n_embd = params['n_embd']
        self.num_factor = params['num_factor']
        self.lr = params['lr']
        
        self.show_config_info()
        
        self.num_user, self.num_item = self.load_representation_data(
            root_path + "u2index.txt",
            root_path + "i2index.txt")
            
        self.path_rep_user = root_path + "user_repr.txt"
        self.path_rep_item = root_path + "item_repr.txt"
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        
            
    def load_representation_data(self, u2index_path, i2index_path):
        count = 0
        with open(u2index_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                count += 1
        num_user = count
        print('Num user: ', num_user)
    
        count = 0
        with open(i2index_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                count += 1
        num_item = count
        print('Num item: ', num_item)
    
        return num_user, num_item
    
    def show_config_info(self):
        print('*'*100)
        print('Num_factor: ', self.num_factor)
        print('Eta: ', self.eta)
        print('Bs: ', self.batch_size)
        print('Lr: ', self.lr)
        print('*'*100)