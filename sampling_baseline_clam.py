import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np


def sample_contrastive_features(pred_df, pt_dir, target_class, sample_classes, total_features=30000):
    """
    From prediction dataframe, sample features from slides predicted as other classes.

    Args:
        pred_df (pd.DataFrame): DataFrame with 'slide_id', 'pred_label', 'true_label', etc.
        pt_dir (str): Directory where .pt feature files are stored.
        target_class (int): Class for which we want contrastive samples.
        sample_classes (list): Other class IDs to sample from.
        total_features (int): Number of features to sample overall.

    Returns:
        sampled_feats: [N, D] tensor of sampled contrastive features.
    """
    # Filter prediction dataframe
    sample_df = pred_df[pred_df['pred_label'].isin(sample_classes)]
    print(f"[INFO] Found {len(sample_df)} slides for sampling from classes {sample_classes}")

    if len(sample_df) == 0:
        raise ValueError("No slides found in the other classes to sample from.")

    # Shuffle and select some slides
    sample_df = sample_df.sample(frac=1, random_state=42)

    sampled_features = []
    current_count = 0
    for idx, row in sample_df.iterrows():
        slide_id = row['slide_id']
        feat_path = os.path.join(pt_dir, f"{slide_id}.pt")

        if not os.path.isfile(feat_path):
            print(f"[WARNING] File not found: {feat_path}")
            continue

        feats = torch.load(feat_path)
        if len(feats.shape) == 1:
            feats = feats.unsqueeze(0)

        # Shuffle and select subset
        feats = feats[torch.randperm(feats.shape[0])]
        remaining = total_features - current_count
        take_n = min(feats.shape[0], remaining)
        sampled_features.append(feats[:take_n])
        current_count += take_n

        print(f"[INFO] Sampled {take_n} from {slide_id}, Total: {current_count}")

        if current_count >= total_features:
            break

    if len(sampled_features) == 0:
        raise RuntimeError("No features collected. Check your prediction file and path settings.")
    sampled_feats = torch.cat(sampled_features, dim=0)
    print(f"[INFO] Final sampled contrastive feature shape: {sampled_feats.shape}")
    return sampled_feats


def main(args):
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{args.fold}.csv')
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    print(f"[INFO] Loaded predictions from fold {args.fold}: {pred_df.shape[0]} samples")

    num_classes = len(args.label_dict)
    for target_class in range(num_classes):
        other_classes = [c for c in range(num_classes) if c != target_class]
        print(f"[INFO] Sampling contrastive features for target class {target_class} vs {other_classes}")

        sampled_feats = sample_contrastive_features(
            pred_df=pred_df,
            pt_dir=args.paths['pt_files'],
            target_class=target_class,
            sample_classes=other_classes,
            total_features=args.total_features
        )
        print(f"[INFO] Sampled features shape: {sampled_feats.shape}")
        
        # save_dir = os.path.join(args.paths['attribution_scores_folder'], 'contrastive_samples')
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f'class_{target_class}_contrastive.pt')
        # torch.save(sampled_feats, save_path)
        # print(f"[INFO] Saved sampled features to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='configs_simea/clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run the model on')
    parser.add_argument('--start_fold', type=int, default=1)
    parser.add_argument('--end_fold', type=int, default=1)
    parser.add_argument('--total_features', type=int, default=30000)

    args = parser.parse_args()

    with open(f'{args.config}', 'r') as f:
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

    print(" > Start computing contrastive samples for dataset:", args.dataset_name)
    for fold_id in range(args.start_fold, args.end_fold + 1):
        args.fold = fold_id
        main(args)
