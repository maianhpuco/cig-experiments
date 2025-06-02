import os
import yaml
import argparse
import torch
import sys 
ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)   
from src.models.clam import load_clam_model
from src.datasets.classification.camelyon16 import return_splits_custom
import numpy as np
import pandas as pd


def main(args):
    fold_id = args.fold
    device = args.device

    print(f"[INFO] Loading model checkpoint from: {args.paths[f'for_ig_checkpoint_path_fold_{fold_id}']}")
    model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device)

    # Load split
    print("[INFO] Loading test set...")
    split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
    label_dict = {'normal': 0, 'tumor': 1}
    _, _, test_dataset = return_splits_custom(
        csv_path=split_csv_path,
        data_dir=args.paths['pt_files'],
        label_dict=label_dict,
        seed=42,
        print_info=True
    )

    model.eval()
    all_preds, all_labels = [], []
    for i in range(len(test_dataset)):
        features, label, _ = test_dataset[i]
        features = features.unsqueeze(0).to(device)  # Add batch dim
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(features)
            pred = torch.argmax(Y_prob, dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

    # Save predictions
    output_df = pd.DataFrame({
        'true_label': all_labels,
        'pred_label': all_preds
    })
    save_path = os.path.join(args.paths['attribution_scores_folder'], f'test_preds_fold{fold_id}.csv')
    output_df.to_csv(save_path, index=False)
    print(f"[INFO] Predictions saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--fold', type=int, default=1, help='Fold index to evaluate')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run the model on')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    if args.dataset_name == 'tcga_renal':
        args.data_dir_map = config['paths']['data_dir']

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    print(" > Start compute IG for dataset: ", args.dataset_name)
    main(args)
