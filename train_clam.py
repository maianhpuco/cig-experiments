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

# Set base path for imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
# sys.path.append(os.path.join(base_path, "src/externals/CLAM"))

# Internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from 
# from dataset_modules.dataset_generic import Generic_MIL_Dataset, return_splits_custom


from dataset_modules.dataset_generic import return_splits_custom
train_dataset, val_dataset, test_dataset = return_splits_custom(
    csv_path="/home/mvu9/processing_datasets/processing_camelyon16/splits/task_1_tumor_vs_normal_100/split_0.csv",
    data_dir="/home/mvu9/processing_datasets/processing_camelyon16/features_fp",
    label_dict={'normal': 0, 'tumor': 1},
    seed=1,
    print_info=True
)
print(len(train_dataset), len(val_dataset), len(test_dataset)) 


def main(args):
    # Load YAML configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set paths from YAML
    args.data_root_dir = cfg['paths']['pt_files']
    args.results_dir = os.path.join(cfg['paths']['save_dir'], 'results')
    args.split_dir = os.path.join(cfg['paths']['save_dir'], 'splits', 'task_1_tumor_vs_normal_100')

    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # Set task and classes
    args.task = 'task_1_tumor_vs_normal'
    args.n_classes = 2
    label_dict = {'normal': 0, 'tumor': 1}

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        # Use custom split loader
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            csv_path=os.path.join(args.split_dir, f'split_{i}.csv'),
            data_dir=args.data_root_dir,
            label_dict=label_dict,
            seed=args.seed,
            print_info=True
        )
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # Write results to pkl
        filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })

    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    settings = {
        'num_splits': args.k,
        'k_start': args.k_start,
        'k_end': args.k_end,
        'task': args.task,
        'max_epochs': args.max_epochs,
        'results_dir': args.results_dir,
        'lr': args.lr,
        'experiment': args.exp_code,
        'reg': args.reg,
        'label_frac': args.label_frac,
        'bag_loss': args.bag_loss,
        'seed': args.seed,
        'model_type': args.model_type,
        'model_size': args.model_size,
        'use_drop_out': args.drop_out,
        'weighted_sample': args.weighted_sample,
        'opt': args.opt,
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B
    }

    print("################# Settings ###################")
    for key, val in settings.items():
        print(f"{key}:  {val}")

    results = main(args)
    print("Finished!")
    print("End script")