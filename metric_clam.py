import os
import argparse
import yaml
import numpy as np
import torch
from torch.nn.functional import softmax
import sys
from tqdm import tqdm

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

def sample_random_features(dataset, num_files=5, feature_dim=1024):
    """Sample random features from the dataset, ensuring shape [N, feature_dim]."""
    indices = np.random.choice(len(dataset), num_files, replace=False)
    feature_list = []
    for idx in indices:
        features, _, _ = dataset[idx]
        features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
        # Ensure features is [N, 1024]
        while features.dim() > 2:
            features = features.squeeze(0)
        if features.dim() != 2 or features.size(1) != feature_dim:
            raise ValueError(f"Expected features shape [N, {feature_dim}], got {features.shape}")
        if features.size(0) > 128:
            indices = torch.randperm(features.size(0))[:128]
            features = features[indices]
        feature_list.append(features)
    # Concatenate features to form a single bag
    concatenated = torch.cat(feature_list, dim=0)
    # Limit to a reasonable bag size
    max_instances = min(concatenated.size(0), 512)
    indices = torch.randperm(concatenated.size(0))[:max_instances]
    return concatenated[indices]

def main(args):
    methods = [
        'contrastive_gradient', 'integrated_gradient', 'vanilla_gradient',
        'expected_gradient', 'integrated_decision_gradient', 'square_integrated_gradient'
    ]

    for fold_id in tqdm(range(args.fold_start, args.fold_end + 1), desc="Processing folds"):
        print(f"Processing Fold {fold_id}")

        split_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        _, _, test_dataset = return_splits_camelyon(
            csv_path=split_path,
            data_dir=args.paths['pt_files'],
            label_dict={'normal': 0, 'tumor': 1},
            seed=args.seed,
            print_info=False,
            use_h5=True
        )

        model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device=args.device)
        model.eval()

        for idx, (features, label, coords) in enumerate(tqdm(test_dataset, desc=f"Processing slides (Fold {fold_id})")):
            basename = test_dataset.slide_data['slide_id'].iloc[idx]
            print(f"\n[{idx+1}/{len(test_dataset)}] Slide: {basename}")

            # Ensure features is [N, D]
            features = features.to(args.device)
            while features.dim() > 2:
                features = features.squeeze(0)
            if features.dim() != 2 or features.size(1) != args.embed_dim:
                print(f"    ⚠️ Skipping slide {basename}: Expected features shape [N, {args.embed_dim}], got {features.shape}")
                continue

            # Generate baseline features
            baseline = sample_random_features(test_dataset, num_files=5, feature_dim=args.embed_dim).to(args.device)
            while baseline.dim() > 2:
                baseline = baseline.squeeze(0)
            if baseline.dim() != 2 or baseline.size(1) != args.embed_dim:
                print(f"    ⚠️ Skipping slide {basename}: Expected baseline shape [N, {args.embed_dim}], got {baseline.shape}")
                continue

            # Compute logits once for features and baseline
            def call_model_function(model, input_tensor, target_class_idx=None):
                while input_tensor.dim() > 2:
                    input_tensor = input_tensor.squeeze(0)
                if input_tensor.dim() != 2:
                    raise ValueError(f"Expected 2D input tensor, got shape {input_tensor.shape}")
                if input_tensor.size(1) != args.embed_dim:
                    raise ValueError(f"Expected feature dim {args.embed_dim}, got {input_tensor.size(1)}")
                with torch.no_grad():
                    logits, _, _ = model(input_tensor.unsqueeze(0))
                return logits

            try:
                features_logits = call_model_function(model, features)
                baseline_logits = call_model_function(model, baseline)
            except Exception as e:
                print(f"    ⚠️ Error computing logits for slide {basename}: {str(e)}")
                continue

            results = []
            for method in tqdm(methods, desc="Computing metrics", leave=False):
                print(f"  - {method}")
                metrics = []

                for cls in range(args.n_classes):
                    score_path = os.path.join(
                        args.paths['attribution_scores_folder'], method,
                        f'fold_{fold_id}', f'class_{cls}', f'{basename}.npy'
                    )
                    if not os.path.exists(score_path):
                        print(f"    ⚠️ Missing {score_path}, skipping.")
                        continue

                    scores = torch.from_numpy(np.load(score_path)).to(args.device)
                    if scores.shape[0] != features.shape[0]:
                        print(f"    ⚠️ Score shape {scores.shape} does not match features shape {features.shape}, skipping.")
                        continue

                    try:
                        aic, sic = compute_aic_and_sic(
                            model, features, baseline, scores, cls,
                            call_model_function=lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50
                        )
                        ins = compute_insertion_auc(
                            model, features, baseline, scores, cls,
                            call_model_function=lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50
                        )
                        dele = compute_deletion_auc(
                            model, features, baseline, scores, cls,
                            call_model_function=lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50
                        )
                        metrics.append((cls, aic, sic, ins, dele))
                    except Exception as e:
                        print(f"    ⚠️ Error computing metrics for class {cls}: {str(e)}")
                        continue

                if metrics:
                    avg_metrics = tuple(np.mean([m[i] for m in metrics]) for i in range(1, 5))
                    results.append((method, *avg_metrics))

            if results:
                ranked = rank_methods(results)
                out_dir = args.paths['metrics_dir']
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, f"clam_fold{fold_id}_{basename}.npy"), np.array(ranked, dtype=object))
                print(f"     Saved: clam_fold{fold_id}_{basename}.npy")

        print(f" Fold {fold_id} complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/clam_camelyon16.yaml')
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=1)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Transfer config parameters to args
    args.dataset_name = config['dataset_name']
    args.paths = config['paths']
    args.n_classes = config.get('n_classes', 2)
    args.drop_out = config.get('drop_out', 0.25)
    args.model_type = config.get('model_type', 'clam_sb')
    args.embed_dim = config.get('embed_dim', 1024)
    args.bag_loss = config.get('bag_loss', 'ce')
    args.model_size = config.get('model_size', 'small')
    args.no_inst_cluster = config.get('no_inst_cluster', False)
    args.inst_loss = config.get('inst_loss', None)
    args.subtyping = config.get('subtyping', False)
    args.bag_weight = config.get('bag_weight', 0.7)
    args.B = config.get('B', 8)
    
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    main(args)