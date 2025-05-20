from __future__ import print_function

import argparse
import os
import math
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

# sys.path.append(base_path)
sys.path.append(os.path.join("src/externals/CLAM")) 

# Internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
# from dataset_modules.dataset_generic import Generic_MIL_Dataset, return_splits_custom
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 
# sys.path.append(base_path) 
# from utils import get_timestamp_str 
from src.datasets.classification.tcga import return_splits_custom 
from datetime import datetime



def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  

def main(args):

    # Load YAML configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    timestamp = get_timestamp_str()
    timestamp = 'final'
    
    # Set paths from YAML
    args.results_dir = os.path.join(cfg['paths']['save_dir'], f'result_{timestamp}_ep{args.max_epochs}')
    # args.split_dir = os.path.join(cfg['paths']['save_dir'], 'splits', 'task_1_tumor_vs_normal_100')
    split_folder =cfg['paths']['split_folder']
    data_dir_map = cfg['paths']['data_dir']
    dataset_name = cfg['dataset_name'] 
    
    
    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # Set task and classes
    args.task = 'task_1_kirp_vs_kirc_vs_kich'
    args.n_classes = 3
    
    label_dict = {'KIRP': 0, 'KIRC': 1, 'KICH': 2}
    
    start = args.k_start if args.k_start != -1 else 1
    end = args.k_end if args.k_end != -1 else args.k -1

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    
    folds = np.arange(start, end+1)
    print("=======Number of folds:", len(folds), "=========")
    start_time = time.time() 
    for i in folds:
        fold_start_time = time.time() 
        print("=======Start fold number:", i, "=========") 
        seed_torch(args.seed)
        if dataset_name == 'tcga': 
            train_csv_path = os.path.join(split_folder, f'fold_{i}/train.csv')
            val_csv_path = os.path.join(split_folder, f'fold_{i}/val.csv')
            test_csv_path = os.path.join(split_folder, f'fold_{i}/test.csv')


            train_dataset, val_dataset, test_dataset = return_splits_custom(
                train_csv_path,
                val_csv_path,
                test_csv_path,
                data_dir_map=data_dir_map,
                label_dict= label_dict,  # This won't affect direct labels
                seed=42,
                print_info=False
            )
        print(f"Train len: {len(train_dataset)} | Val len: {len(val_dataset)} | Test len: {len(test_dataset)}")
          
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # Write results to pkl
        
        fold_result_dir = os.path.join(args.results_dir, f'fold{i}')
        os.makedirs(fold_result_dir, exist_ok=True)

        filename = os.path.join(fold_result_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)
        
        print(f"======> [x] Saved checkpoint for fold {i} at: {filename}")  
        
        # filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        # save_pkl(filename, results)
        # print(f"======> [x] Saved checkpoint for fold {i} at: {filename}") 
        fold_duration = time.time() - fold_start_time
        print(f" --> Fold {i} finished in {fold_duration:.2f} seconds") 
        
    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })
    
    print("=======Number of folds:", len(folds), "=========") 
    
    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    print(f"=========> [x] Summary saved at: {os.path.join(args.results_dir, save_name)}")
    total_duration = time.time() - start_time

    print(f"\n ----> All folds completed in {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)") 

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_frac', type=float, default=1.0)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--log_data', action='store_true', default=False)
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--exp_code', type=str, default='clam_camelyon16')
    parser.add_argument('--weighted_sample', action='store_true', default=False)
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    parser.add_argument('--no_inst_cluster', action='store_true', default=False)
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
    parser.add_argument('--subtyping', action='store_true', default=False)
    parser.add_argument('--bag_weight', type=float, default=0.7)
    parser.add_argument('--B', type=int, default=8)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge YAML config into args
    for key, value in config.items():
        setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# Settings ###################")
    settings = vars(args).copy()
    settings.pop('paths', None)
    for key, val in settings.items():
        print(f"{key}: {val}")
    print(f"[INFO] Max Epochs set to: {args.max_epochs}")
    main(args)
    
    
    