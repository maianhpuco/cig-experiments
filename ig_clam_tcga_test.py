'''
this code is to test ig's model on tcga-renal + clam
'''
import os
import sys 
import argparse
import time
import numpy as np
import h5py
import torch
import yaml
import pickle
from torch import nn
import shutil

ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)  

from clam import load_clam_model  
from attr_method._common import (
    call_model_function
) 
# from src.datasets.classification.camelyon16 import return_splits_custom
# from utils.utils import load_config
from src.datasets.classification.tcga import return_splits_custom  


def sample_random_features(dataset, num_files=20):
    """
    Randomly sample feature arrays from the dataset and stack them.
    Handles variable-length inputs by selecting a fixed number of patches from each.
    """
    indices = np.random.choice(len(dataset), num_files, replace=False)
    feature_list = []
    selected_ids = []

    for idx in indices:
        features, _, _ = dataset[idx]  # Unpack three elements
        features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
        if features.size(0) > 128:
            features = features[:128]
        feature_list.append(features)
        selected_ids.append(dataset.slide_data['slide_id'].iloc[idx])

    padded = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
    flattened = padded.view(-1, padded.size(-1))
    return flattened, selected_ids


def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    args = parser.parse_args(args=[])  # empty args for testing
    return args

def main(args):
    if args.ig_name == 'integrated_gradient':
        from attr_method.integrated_gradient import IntegratedGradients as AttrMethod
    elif args.ig_name == 'vanilla_gradient':
        from attr_method.vanilla_gradient import VanillaGradients as AttrMethod
    elif args.ig_name == 'contrastive_gradient':
        from attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod
    elif args.ig_name == 'expected_gradient':
        from attr_method.expected_gradient import ExpectedGradients as AttrMethod
    elif args.ig_name == 'integrated_decision_gradient':
        from attr_method.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod
    elif args.ig_name == 'optim_square_integrated_gradient':
        from attr_method.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    elif args.ig_name == 'square_integrated_gradient':
        from attr_method.square_integrated_gradient import SquareIntegratedGradients as AttrMethod

    print(f"Running for {args.ig_name} Attribution method")

    attribution_method = AttrMethod()

 
    # Create directory for memmap files
    memmap_path = os.path.join(args.paths['memmap_path'], f'{args.ig_name}')
    if os.path.exists(memmap_path):
        shutil.rmtree(memmap_path)  # Remove if exists
    os.makedirs(memmap_path, exist_ok=True) 
    
    # =========== config for camelyon16 ============ 
    if args.dataset_name == 'camelyon16':   
        split_csv_path = os.path.join(args.paths['split_folder'], 'fold_1.csv') 
        train_dataset, _, test_dataset = return_splits_custom(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict={'normal': 0, 'tumor': 1},
            seed=args.seed,
            print_info=False, 
            use_h5=True
        )
        print("-- Total number of sample in test set:", len(test_dataset))
    
    # =========== config for tgca renal ============  
    elif args.dataset_name=='tcga_renal':
        split_folder = args.paths['split_folder']
        data_dir_map = {
            'KICH': args.paths['data_dir']['kich'],
            'KIRP': args.paths['data_dir']['kirp'],
            'KIRC': args.paths['data_dir']['kirc'],
        }
        label_dict = {'KICH': 0, 'KIRP': 1, 'KIRC': 2}
        
            
        for fold_id in range(1, 2):  # e.g., loop over 1 fold only for now
            print(f"Processing Fold {fold_id}")
            
                
            train_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
            val_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
            test_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

            train_dataset, val_dataset, test_dataset = return_splits_custom(
                            train_csv_path,
                            val_csv_path,
                            test_csv_path,
                            data_dir_map=data_dir_map,
                            label_dict= label_dict,  # This won't affect direct labels
                            seed=42,
                            print_info=False
                        )
            print("-- Total number of samples in test set:", len(test_dataset)) 
            
            
            # ====== load the clam model's weight base in the fold id ======
            model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device=args.device)
            # ====== load the clam model's weight base in the fold id ====== 
            
             
        for idx, (features, label, coords) in enumerate(test_dataset):
            # print("- Feature shape", features.shape)
            # print("- label", label)
            # print("- coords", coords)
            basename = test_dataset.slide_data['slide_id'].iloc[idx]
            
            # print("basename", basename)
            print(f"Processing the file number {idx+1}/{len(test_dataset)}")
            
            start = time.time()

            stacked_features_baseline, _ = sample_random_features(test_dataset, num_files=20)
            print("stack features ", stacked_features_baseline.shape)
            
            stacked_features_baseline = (stacked_features_baseline.to(args.device, dtype=torch.float32) 
                                        if isinstance(stacked_features_baseline, torch.Tensor) 
                                        else torch.tensor(stacked_features_baseline, dtype=torch.float32, device=args.device))
            
            kwargs = {
                "x_value": features,
                "call_model_function": call_model_function,
                "model": model,
                "baseline_features": stacked_features_baseline,
                "memmap_path": memmap_path,
                "x_steps": 50,
                "device": args.device, 
                "call_model_args": {"target_class_idx": int(label)}
            }

            attribution_values = attribution_method.GetMask(**kwargs)
            scores = attribution_values.mean(1)
            print("- Score result shape: ", scores.shape)
            # ==========saving the score result ============== 
              # Create directory for attribution scores
            score_save_path = os.path.join(args.paths['attribution_scores_folder'], f'{args.ig_name}', f'fold_{i}')
            if os.path.exists(score_save_path):
                shutil.rmtree(score_save_path)  # Remove if exists
            os.makedirs(score_save_path, exist_ok=True)
 
            # print("- Score result mean: ", scores.mean())
            _save_path = os.path.join(score_save_path, f'{basename}.npy')
            # np.save(_save_path, scores.detach().cpu().numpy())
            # np.save(_save_path, scores)
            
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            np.save(_save_path, scores)

            print(f"Done save result numpy file at shape {scores.shape} at {_save_path}")
            # ==========saving the score result ==============  
            
            
            break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--ig_name',
                        default='integrated_gradient',
                        choices=[
                            'integrated_gradient',
                            'expected_gradient',
                            'integrated_decision_gradient',
                            'contrastive_gradient',
                            'vanilla_gradient',
                            'square_integrated_gradient',
                            'optim_square_integrated_gradient'
                        ],
                        help='Choose the attribution method to use.')
    args = parser.parse_args()

    with open(f'{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)
            
    args.dataset_name = config['dataset_name']
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = "cpu"
    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)
    
    print(" > Start compute IG for dataset: ", args.dataset_name)
    main(args)

        # CLAM model 
        # logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True) 