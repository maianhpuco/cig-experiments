import os
import argparse
import yaml
import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
import sys
import pandas as pd

import warnings
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
 
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(clf_path)

from src.datasets.classification.tcga import return_splits_custom
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon
from clam import load_clam_model
from src.metrics import (
    compute_aic_and_sic,
    compute_insertion_auc,
    compute_deletion_auc,
    rank_methods
)

def sample_random_features(dataset, feature_dim=1024):
    idx = np.random.randint(0, len(dataset))
    features, _, _ = dataset[idx]
    features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
    features = features.squeeze() if features.dim() > 2 else features
    if features.dim() != 2 or features.size(1) != feature_dim:
        raise ValueError(f"Invalid feature shape: {features.shape}")
    if features.size(0) > 32:
        features = features[torch.randperm(features.size(0))[:32]]
    return features

def call_model_function(model, input_tensor, target_class_idx=None):
    if input_tensor.dim() != 2:
        raise ValueError(f"Expected input shape [N, D], got {input_tensor.shape}")
    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, tuple):
            logits = out[0]
            return logits
        raise RuntimeError("Unexpected model output structure")

def main(args):
    all_results = []
    for fold_id in tqdm(range(args.fold_start, args.fold_end + 1), desc="Processing folds"):
        print(f"Processing Fold {fold_id}")
        split_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        _, _, test_dataset = return_splits_camelyon(
            csv_path=split_path,
            data_dir=args.paths['pt_files'],
            label_dict=args.label_dict,
            seed=args.seed,
            print_info=False,
            use_h5=True
        )

        model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device=args.device)
        model.eval()

        for method in args.methods:
            print(f"\n=========================> Method: {method}")
            # for cls in range(args.n_classes):
            #     print(f"Class {cls}")
            class_score_folder_class_0 = os.path.join(
                args.paths['attribution_scores_folder'], method,
                f'fold_{fold_id}', f'class_0'
            )
            class_score_folder_class_1 = os.path.join(
                args.paths['attribution_scores_folder'], method,
                f'fold_{fold_id}', f'class_1'
            )
            
            count = 0 
            for idx, (features, label, coords) in enumerate(test_dataset):
                if count >= 0:
                    break 
                count += 1 
                
                slide_id = test_dataset.slide_data['slide_id'].iloc[idx]
                score_path_class_0 = os.path.join(class_score_folder_class_0, f"{slide_id}.npy")
                score_path_class_1 = os.path.join(class_score_folder_class_1, f"{slide_id}.npy")
            
                scores_class_0 = np.load(score_path_class_0)
                scores_class_1 = np.load(score_path_class_1) 


                if scores_class_0.shape != scores_class_1.shape:
                    print(f"⚠️ Mismatched shape for {slide_id}: {scores_class_0.shape} vs {scores_class_1.shape}")
                    continue

                # Compute metrics
                mean_diff = np.mean(np.abs(scores_class_0 - scores_class_1))
                cos_sim = 1 - cosine(scores_class_0, scores_class_1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pear_corr, _ = pearsonr(scores_class_0, scores_class_1)

                print(f"Slide: {slide_id}")
                print(f"  - Mean abs diff   : {mean_diff:.4f}")
                print(f"  - Cosine similarity: {cos_sim:.4f}")
                print(f"  - Pearson corr     : {pear_corr:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs_simea/clam_camelyon16.yaml')
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=1)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.label_dict = {int(k): v for k, v in config['label_dict'].items()}
 
    # model configs 
    args.dataset_name = config['dataset_name']
    args.paths = config['paths']
    args.n_classes = config.get('n_classes', 2)
    args.drop_out = config.get('drop_out', 0.25)
    args.model_type = config.get('model_type', 'clam_sb')
    args.embed_dim = config.get('embed_dim', 1024)
    args.bag_loss = config.get('bag_loss', 'ce')
    args.model_size = config.get('model_size', 'small')
    args.no_inst_cluster = config.get('no_inst_cluster', True)
    args.inst_loss = config.get('inst_loss', None)
    args.subtyping = config.get('subtyping', False)
    args.bag_weight = config.get('bag_weight', 0.7)
    args.B = config.get('B', 1)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # ig configs
    args.methods = [
        'contrastive_gradient', 'integrated_gradient', 'vanilla_gradient',
        'expected_gradient', 'integrated_decision_gradient', 'square_integrated_gradient'
    ] 
    # args.methods = ['integrated_gradient']
    main(args)