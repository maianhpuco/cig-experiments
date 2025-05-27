import os
import argparse
import yaml
import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
import sys

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
    if features.dim() > 2:
        features = features.squeeze()
    if features.dim() != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    if features.size(1) != feature_dim:
        raise ValueError(f"Expected feature dim {feature_dim}, got {features.size(1)}")
    max_patches = 32
    if features.size(0) > max_patches:
        indices = torch.randperm(features.size(0))[:max_patches]
        features = features[indices]
    return features

def call_model_function(model, input_tensor, args):
    if input_tensor.dim() > 2:
        input_tensor = input_tensor.squeeze()
    if input_tensor.dim() != 2:
        raise ValueError(f"Expected 2D input tensor, got shape {input_tensor.shape}")
    if input_tensor.size(1) != args.embed_dim:
        raise ValueError(f"Expected feature dim {args.embed_dim}, got {input_tensor.size(1)}")
    try:
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            if isinstance(output, (list, tuple)) and len(output) == 5:
                logits, _, _, _, _ = output
            elif isinstance(output, (list, tuple)) and len(output) == 3:
                logits, _, _ = output
            else:
                raise RuntimeError(f"Unexpected number of outputs from model: {len(output)}")
        return logits
    except Exception as e:
        raise RuntimeError(f"Model forward failed: {str(e)}")

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

        for idx, (features, label, coords) in enumerate(tqdm(test_dataset, desc=f"Slides Fold {fold_id}")):
            basename = test_dataset.slide_data['slide_id'].iloc[idx]

            features = features.to(args.device)
            if features.dim() > 2:
                features = features.squeeze()
            if features.dim() != 2 or features.size(1) != args.embed_dim:
                continue
            if features.size(0) > 32:
                features = features[torch.randperm(features.size(0))[:32]]
            features = (features - features.mean()) / (features.std() + 1e-8)

            baseline = sample_random_features(test_dataset, args.embed_dim).to(args.device)
            baseline = (baseline - baseline.mean()) / (baseline.std() + 1e-8)

            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            if torch.isnan(baseline).any() or torch.isinf(baseline).any():
                continue
            if baseline.size(0) != features.size(0):
                continue

            try:
                features_logits = call_model_function(model, features, args)
                baseline_logits = call_model_function(model, baseline, args)
            except Exception as e:
                print(f"⚠️ Model forward failed for slide {basename}: {str(e)}")
                continue

            results = []
            for method in tqdm(methods, desc="Metrics", leave=False):
                metrics = []
                for cls in range(args.n_classes):
                    score_path = os.path.join(
                        args.paths['attribution_scores_folder'], method,
                        f'fold_{fold_id}', f'class_{cls}', f'{basename}.npy'
                    )
                    if not os.path.exists(score_path):
                        continue
                    scores = torch.from_numpy(np.load(score_path)).to(args.device)
                    if scores.shape[0] != features.shape[0]:
                        continue
                    try:
                        aic, sic = compute_aic_and_sic(model, features, baseline, scores, cls,
                            lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50)
                        ins = compute_insertion_auc(model, features, baseline, scores, cls,
                            lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50)
                        dele = compute_deletion_auc(model, features, baseline, scores, cls,
                            lambda m, x, target_class_idx=None: features_logits if torch.equal(x, features) else baseline_logits,
                            steps=50)
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

        print(f"Fold {fold_id} complete.")

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

    main(args)
