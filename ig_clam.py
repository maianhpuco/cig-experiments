'''
this code is to test ig's model on tcga-renal + clam
'''
import os
import sys 
import argparse
import time
import numpy as np
import torch
import yaml
import shutil
from torch import nn

ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)  

from clam import load_clam_model  
from attr_method_tcga_renal._common import call_model_function
from src.datasets.classification.tcga import return_splits_custom  
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon

def sample_random_features(dataset, num_files=20):
    indices = np.random.choice(len(dataset), num_files, replace=False)
    feature_list = []
    for idx in indices:
        features, _, _ = dataset[idx]
        features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
        if features.size(0) > 128:
            features = features[:128]
        feature_list.append(features)
    padded = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
    flattened = padded.view(-1, padded.size(-1))
    return flattened

def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])

def main(args):
    if args.ig_name == 'integrated_gradient':
        from attr_method_tcga_renal.integrated_gradient import IntegratedGradients as AttrMethod
    elif args.ig_name == 'vanilla_gradient':
        from attr_method_tcga_renal.vanilla_gradient import VanillaGradients as AttrMethod
    elif args.ig_name == 'contrastive_gradient':
        from attr_method_tcga_renal.contrastive_gradient import ContrastiveGradients as AttrMethod
    elif args.ig_name == 'expected_gradient':
        from attr_method_tcga_renal.expected_gradient import ExpectedGradients as AttrMethod
    elif args.ig_name == 'integrated_decision_gradient':
        from attr_method_tcga_renal.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod
    elif args.ig_name == 'optim_square_integrated_gradient':
        from attr_method_tcga_renal.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    elif args.ig_name == 'square_integrated_gradient':
        from attr_method_tcga_renal.square_integrated_gradient import SquareIntegratedGradients as AttrMethod

    print(f"Running for {args.ig_name} Attribution method")
    attribution_method = AttrMethod()

    memmap_path = os.path.join(args.paths['memmap_path'], f'{args.ig_name}')
    if os.path.exists(memmap_path):
        shutil.rmtree(memmap_path)
    os.makedirs(memmap_path, exist_ok=True)

    if args.dataset_name == 'tcga_renal':
        split_folder = args.paths['split_folder']
        data_dir_map = {
            'KICH': args.paths['data_dir']['kich'],
            'KIRP': args.paths['data_dir']['kirp'],
            'KIRC': args.paths['data_dir']['kirc'],
        }
        label_dict = {'KICH': 0, 'KIRP': 1, 'KIRC': 2}

        for fold_id in range(1, 2):
            print(f"Processing Fold {fold_id}")

            train_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
            val_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
            test_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

            train_dataset, val_dataset, test_dataset = return_splits_custom(
                train_csv_path,
                val_csv_path,
                test_csv_path,
                data_dir_map=args.data_dir_map,
                label_dict=label_dict,
                seed=42,
                print_info=False
            )
            print("-- Total number of samples in test set:", len(test_dataset))
            args.n_classes = 3
            model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device=args.device)

            for idx, (features, label, coords) in enumerate(test_dataset):
                basename = test_dataset.slide_data['slide_id'].iloc[idx]
                print(f"\nProcessing file {idx + 1}/{len(test_dataset)}: {basename}")

                features = features.to(args.device, dtype=torch.float32)
                stacked_features_baseline = sample_random_features(test_dataset).to(args.device, dtype=torch.float32)

                for class_idx in range(args.n_classes):
                    print(f"⮕ Attribution for class {class_idx}")
                    kwargs = {
                        "x_value": features,
                        "call_model_function": call_model_function,
                        "model": model,
                        "baseline_features": stacked_features_baseline,
                        "memmap_path": memmap_path,
                        "x_steps": 50,
                        "device": args.device,
                        "call_model_args": {"target_class_idx": class_idx}
                    }

                    attribution_values = attribution_method.GetMask(**kwargs)
                    scores = attribution_values.mean(1)
                    print(f"- Score shape: {scores.shape}")

                    score_save_path = os.path.join(
                        args.paths['attribution_scores_folder'], f'{args.ig_name}', f'fold_{fold_id}', f'class_{class_idx}'
                    )
                    os.makedirs(score_save_path, exist_ok=True)
                    save_path = os.path.join(score_save_path, f'{basename}.npy')

                    if isinstance(scores, torch.Tensor):
                        scores = scores.detach().cpu().numpy()
                    np.save(save_path, scores)

                    print(f"✅ Saved scores for {args.dataset_name},  {fold_id} class {class_idx} at {save_path}")

                break

    elif args.dataset_name == 'camelyon16':
        for fold_id in range(1, 2):
            split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
            train_dataset, _, test_dataset = return_splits_camelyon(
                csv_path=split_csv_path,
                data_dir=args.paths['pt_files'],
                label_dict={'normal': 0, 'tumor': 1},
                seed=args.seed,
                print_info=False,
                use_h5=True
            )
            print("-- Total number of sample in test set:", len(test_dataset))
            args.n_classes = 2 
            # model = load_clam_model(args, args.paths['for_ig_checkpoint_path'], device=args.device)
            model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device=args.device)
            for idx, (features, label, coords) in enumerate(test_dataset):
                basename = test_dataset.slide_data['slide_id'].iloc[idx]
                print(f"\nProcessing file {idx + 1}/{len(test_dataset)}: {basename}")

                features = features.to(args.device, dtype=torch.float32)
                stacked_features_baseline = sample_random_features(test_dataset).to(args.device, dtype=torch.float32)

                for class_idx in range(args.n_classes):
                    print(f"⮕ Attribution for class {class_idx}")
                    kwargs = {
                        "x_value": features,
                        "call_model_function": call_model_function,
                        "model": model,
                        "baseline_features": stacked_features_baseline,
                        "memmap_path": memmap_path,
                        "x_steps": 50,
                        "device": args.device,
                        "call_model_args": {"target_class_idx": class_idx}
                    }
                    attribution_values = attribution_method.GetMask(**kwargs)
                    scores = attribution_values.mean(1)
                    print(f"- Score shape: {scores.shape}")

                    score_save_path = os.path.join(
                        args.paths['attribution_scores_folder'], f'{args.ig_name}', f'fold_{fold_id}', f'class_{class_idx}'
                    )
                    os.makedirs(score_save_path, exist_ok=True)
                    save_path = os.path.join(score_save_path, f'{basename}.npy')

                    if isinstance(scores, torch.Tensor):
                        scores = scores.detach().cpu().numpy()
                    np.save(save_path, scores)

                    print(f"✅ Saved scores for {args.dataset_name},  {fold_id} class {class_idx} at {save_path}")

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
                        ])
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run the model on')

    args = parser.parse_args()

    with open(f'{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    if args.dataset_name =='tcga_renal':
        args.data_dir_map = config['paths']['data_dir'] 
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    print(" > Start compute IG for dataset: ", args.dataset_name)
    main(args)
