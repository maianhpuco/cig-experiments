from __future__ import print_function
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
import sys
import time
from datetime import datetime
import pickle
import os 
# Add model path
sys.path.append(os.path.abspath(os.path.join("src/models/classifiers")))

# Imports
from mlp_classifier import load_model_mlp
from mlp_trainer import train 

# from trainer import seed_torch
from utils.file_utils import save_pkl

from src.datasets.classification.tcga import return_splits_custom  as return_splits_tcga 
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16 

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

def load_dataset(args):
    fold_id = args.fold 
    
    if args.dataset_name == 'camelyon16':
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}

        _, _, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        print(f"[INFO] Test Set Size: {len(test_dataset)}")

    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        split_folder = args.paths['split_folder']
        data_dir_map = args.paths['data_dir']

        train_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
        val_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
        test_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

        train_dataset, val_dataset, test_dataset = return_splits_tcga(
            train_csv_path,
            val_csv_path,
            test_csv_path,
            data_dir_map=data_dir_map,
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
       
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print(f"[INFO] FOLD {fold_id} -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}") 
    
    return train_dataset, val_dataset, test_dataset 



def save_pkl(filename, data):
    """
    Save a Python object to a .pkl file.

    Args:
        filename (str): Path to the output .pkl file.
        data (Any): Python object to save.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved to: {filename}")
     
def main(args):

    args.dataset_name = args.dataset_name
    args.n_classes = args.n_classes
    label_dict = args.label_dict 

    args.results_dir = args.paths['result_dir']
    os.makedirs(args.results_dir, exist_ok=True)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []

    fold_dir = os.path.join(args.results_dir, f'fold_{args.fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting fold {args.fold}, saving results to: {fold_dir}")
    
    seed_torch(args.seed)
    args.results_dir = fold_dir
    train_dataset, val_dataset, test_dataset = load_dataset(args) 
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    datasets = (train_dataset, val_dataset, test_dataset)

    results, test_auc, val_auc, test_acc, val_acc = train(datasets, args)

    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)

    save_pkl(os.path.join(fold_dir, f'split_{args.fold}_results.pkl'), results)

    summary_df = pd.DataFrame({
        'folds': fold,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })
    summary_csv = os.path.join(args.results_dir, f'summary_partial.csv')
    summary_df.to_csv(summary_csv)
    print(f"[INFO] Summary saved to: {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WSI Training with MLP_Classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    # parser.add_argument('--ckpt_path', type=str, default=None)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        if key != 'paths':
            setattr(args, key, val)
    args.paths = config['paths']

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("################# Settings ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print(f"[INFO] Max Epochs set to: {args.max_epochs}")
    for fold in range(args.k_start, args.k_end+1):
        print(f"Training Fold: {fold}")
        args.fold = fold
        main(args)
