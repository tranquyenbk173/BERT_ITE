from model import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model_Bert_Ite_both_cat')
    # Arguments
    parser.add_argument('--num_factor', type=int, default=64, help='(default=%(default)d)', required=True)
    parser.add_argument('--eta', default=0.5, type=float, required=True, help='(default=%(default)s)')
    parser.add_argument('--batch_size', default=512, type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--lr', default=0.002, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--dataset', default='tmall', type=str, required=True, 
                                        choices = ['tmall', 'retail_rocket', 'recobell'], help='(default=%(default)s)')
    args=parser.parse_args()

    root_path = '/content/drive/MyDrive/ITE/ITE_code/site_data/' + args.dataset + '/'
    save_path_name='model_bertIte_both_cat'
    log_path = root_path + 'log/{}/'.format(save_path_name)
    file_model = root_path + 'saved_model/{}/'.format(save_path_name)
    save_log = True
    save_model = True
    
    if not os.path.isdir(log_path):
            os.mkdir(log_path)
            print('Make dir----', log_path)
            
    if not os.path.isdir(file_model):
            os.mkdir(file_model)
            print('Make dir----', file_model)
    
    params = {
        'attn_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'n_head': 2,
        'n_layer': 1,
        'interval': 64,
        'eta': args.eta,
        'reg_lambda': 0.005,
        'num_neg': 9,
        'max_len': 40,
        'eval_top_k': [5, 10, 20, 30, 40, 50],
        'batch_size': args.batch_size,  
        'n_embd': args.num_factor,
        'num_factor': args.num_factor,
        'lr': args.lr
    }
    model = Manager(root_path=root_path, params=params, log_path=log_path, saved_model_path=file_model, restore=False)
    model.train_and_evaluate()
    