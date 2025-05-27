import os
import argparse
import yaml
import numpy as np
import torch
from torch.nn.functional import softmax
import sys
from tqdm import tqdm

# Get the absolute path of the parent of the parent directory
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

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.dataset_name = config['dataset_name']
    args.paths = config['paths']
    args.n_classes = 2
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

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

            features = features.to(args.device)
            baseline = sample_random_features(test_dataset).to(args.device)

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

                    aic, sic = compute_aic_and_sic(
                        model, features, baseline, scores, cls,
                        call_model_function=lambda m, x, target_class_idx: m(x, target_class_idx=target_class_idx),
                        steps=100
                    )
                    ins = compute_insertion_auc(
                        model, features, baseline, scores, cls,
                        call_model_function=lambda m, x, target_class_idx: m(x, target_class_idx=target_class_idx),
                        steps=100
                    )
                    dele = compute_deletion_auc(
                        model, features, baseline, scores, cls,
                        call_model_function=lambda m, x, target_class_idx: m(x, target_class_idx=target_class_idx),
                        steps=100
                    )

                    metrics.append((cls, aic, sic, ins, dele))

                if metrics:
                    avg_metrics = tuple(np.mean([m[i] for m in metrics]) for i in range(1, 5))
                    results.append((method, *avg_metrics))

            if results:
                ranked = rank_methods(results)
                out_dir = args.paths['metrics_dir']
                os.makedirs(out_dir, exist_ok=True)

                np.save(os.path.join(out_dir, f"clam_fold{fold_id}_{basename}.npy"), np.array(ranked, dtype=object))
                print(f"     Saved: clam_fold{fold_id}_{basename}.npy")

        # Optional: aggregate fold result summary
        print(f" Fold {fold_id} complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/clam_camelyon16.yaml')
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=1)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    main(args)