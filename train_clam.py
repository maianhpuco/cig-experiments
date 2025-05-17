from __future__ import print_function
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

# Set base path for imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.append(base_path)
sys.path.append(os.path.join("src/externals/CLAM"))
# Internal imports

from src.datasets.classification.camelyon16 import return_splits_custom
from utils.file_utils import save_pkl
from utils.utils import seed_torch
from utils.core_utils import train 


split_csv_path = "./camelyon16_csv_splits_camil/splits_test.csv" 

train_dataset, val_dataset, val_dataset = return_splits_custom(
    csv_path=split_csv_path,
    data_dir='/home/mvu9/processing_datasets/processing_camelyon16/features_fp',
    label_dict={'normal': 0, 'tumor': 1},  # This won't affect direct labels
    seed=42,
    print_info=True
)
 
print(len(train_dataset), len(val_dataset), len(val_dataset)) 
 

 
def main(args):
    # Set paths
    args.data_root_dir = args.paths['pt_files']
    args.results_dir = os.path.join(args.paths['save_dir'], 'results')
    args.split_dir = os.path.join(args.paths['save_dir'], 'splits', 'task_1_tumor_vs_normal_100')

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Set task and classes
    args.task = 'task_1_tumor_vs_normal'
    args.n_classes = 2
    label_dict = {'normal': 0, 'tumor': 1}

    # Single split training
    seed_torch(args.seed)
    train_dataset, val_dataset, test_dataset = return_splits_custom(
        csv_path=os.path.join(args.split_dir, 'split_0.csv'),
        data_dir=args.data_root_dir,
        label_dict=label_dict,
        seed=args.seed,
        print_info=True
    )
    
    datasets = (train_dataset, val_dataset, test_dataset)
    print("=============== Start running the CLAM training=================")
    results, test_auc, val_auc, test_acc, val_acc = train(datasets, 0, args)
    
    # Save results
    print("=============== Start running the CLAM training=================") 
    filename = os.path.join(args.results_dir, 'split_0_results.pkl')
    print("Result will be save to: ", filename)
    
    save_pkl(filename, results)
    
    final_df = pd.DataFrame({
        'folds': [0],
        'test_auc': [test_auc],
        'val_auc': [val_auc],
        'test_acc': [test_acc],
        'val_acc': [val_acc]
    })

    print(final_df)
    final_df.to_csv(os.path.join(args.results_dir, 'summary.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    # Load YAML configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config to argparse.Namespace
    from argparse import Namespace
    args = Namespace(**config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# Settings ###################")
    settings = vars(args).copy()
    settings.pop('paths', None)  # Exclude paths for cleaner output
    for key, val in settings.items():
        print(f"{key}: {val}")

    main(args)
    print("Finished!")