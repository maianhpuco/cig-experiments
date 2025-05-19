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

ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)  

from clam import load_clam_model  
from attr_method._common import (
    sample_random_features,
    call_model_function
) 
from src.datasets.classification.camelyon16 import return_splits_custom
# from utils.utils import load_config

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

    score_save_path = os.path.join(args.paths['attribution_scores_folder'], f'{args.ig_name}')
    os.makedirs(score_save_path, exist_ok=True)

    model = load_clam_model(args, args.paths['for_ig_checkpoint_path'], device=args.device)

    split_csv_path = os.path.join(args.paths['split_folder'], 'fold_1.csv')
    train_dataset, _, test_dataset = return_splits_custom(
        csv_path=split_csv_path,
        data_dir=args.paths['pt_files'],
        label_dict={'normal': 0, 'tumor': 1},
        seed=args.seed,
        print_info=False
    )

    # if args.do_normalizing:
    #     print("[INFO] Recomputing mean and std from train set")
    #     # Aggregate all feature rows from train set
    #     all_feats = []
    #     for feats, _ in train_dataset:
    #         feats = np.atleast_2d(feats)
    #         all_feats.append(feats)
    #     all_feats = np.concatenate(all_feats, axis=0)
    #     mean = all_feats.mean(axis=0)
    #     std = all_feats.std(axis=0)

    # print(">>>>>>>>>>>----- Total number of sample in test set:", len(test_dataset))

    for idx, (features, label, basename) in enumerate(test_dataset):
        print(f"Processing the file number {idx+1}/{len(test_dataset)}")
        start = time.time()

        # if args.do_normalizing:
        #     print("----- normalizing")
        #     features = (features - mean) / (std + 1e-8)

        stacked_features_baseline, _ = sample_random_features(test_dataset, num_files=20)
        stacked_features_baseline = stacked_features_baseline.numpy()

        kwargs = {
            "x_value": features,
            "call_model_function": call_model_function,
            "model": model,
            "baseline_features": stacked_features_baseline,
            "memmap_path": args.memmap_path,
            "x_steps": 50,
        }

        attribution_values = attribution_method.GetMask(**kwargs)
        scores = attribution_values.mean(1)
        _save_path = os.path.join(score_save_path, f'{basename}.npy')
        np.save(_save_path, scores)
        print(f"Done save result numpy file at shape {scores.shape} at {_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='clam_camelyon16.yaml')
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

    with open(f'./configs_simea/{args.config_file}', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.do_normalizing = True

    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    main(args)
