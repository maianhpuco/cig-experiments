import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def preload_contrastive_features(dataset, slide_ids, max_workers=4):
    """Preload features for given slide IDs using multithreading."""
    def load_features(slide_id):
        try:
            feats = dataset.get_features_by_slide_id(slide_id)
            return slide_id, feats
        except Exception as e:
            print(f"[WARNING] Could not load slide {slide_id}: {e}")
            return slide_id, None

    features_cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_slide = {executor.submit(load_features, slide_id): slide_id for slide_id in slide_ids}
        for future in as_completed(future_to_slide):
            slide_id, feats = future.result()
            if feats is not None:
                features_cache[slide_id] = feats
    return features_cache

def sample_contrastive_features(pred_df, dataset, target_slide_id, target_label, sample_classes, max_slides=30, features_cache=None):
    """Sample contrastive features to match target slide shape."""
    # Get target slide features
    target_feats = dataset.get_features_by_slide_id(target_slide_id)
    num_target_feats, feat_dim = target_feats.shape  # [N, D]
    
    target_pred = pred_df.loc[pred_df['slide_id'] == target_slide_id, 'pred_label'].values[0]
    print(f"[INFO] Target slide: {target_slide_id} | Predicted Label: {target_pred} | #Features: {num_target_feats} | Dim: {feat_dim}")

    # Use preloaded features or load on-the-fly
    if features_cache is None:
        sample_df = pred_df[pred_df['pred_label'].isin(sample_classes)].sample(frac=1, random_state=42)[:max_slides]
        if len(sample_df) == 0:
            raise ValueError(f"No contrastive samples for classes {sample_classes}")
        slide_ids = sample_df['slide_id'].tolist()
        features_cache = preload_contrastive_features(dataset, slide_ids, max_workers=4)

    # Collect valid features
    sampled_features = []
    total_collected = 0
    for slide_id in list(features_cache.keys())[:max_slides]:
        feats = features_cache[slide_id]
        pred_label = pred_df.loc[pred_df['slide_id'] == slide_id, 'pred_label'].values[0]
        sampled_features.append(feats)
        total_collected += feats.shape[0]
        print(f"[INFO] Using slide: {slide_id} | Predicted Label: {pred_label} | Features: {feats.shape[0]}")

    if total_collected == 0:
        raise RuntimeError("No contrastive features collected")

    # Concatenate and adjust features
    all_feats = torch.cat(sampled_features, dim=0)
    print(f"[INFO] Collected features: {total_collected} | Shape: {all_feats.shape}")

    if total_collected >= num_target_feats:
        indices = torch.randperm(total_collected, device='cpu')[:num_target_feats]
        sampled_feats = all_feats[indices]
        print(f"[INFO] Trimmed to {num_target_feats} features")
    else:
        shortage = num_target_feats - total_collected
        repeat_indices = torch.randint(0, total_collected, (shortage,), device='cpu')
        repeated_feats = all_feats[repeat_indices]
        sampled_feats = torch.cat([all_feats, repeated_feats], dim=0)
        print(f"[INFO] Padded with {shortage} repeated features")

    assert sampled_feats.shape == target_feats.shape, (
        f"Shape mismatch: sampled_feats {sampled_feats.shape} != target_feats {target_feats.shape}"
    )
    return sampled_feats

def load_dataset(args):
    if args.dataset_name == 'camelyon16':
        from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{args.fold}.csv')
        args.label_dict = {'normal': 0, 'tumor': 1}
        _, _, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=args.label_dict,
            seed=42,
            print_info=False
        )
        return test_dataset

    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        split_folder = args.paths['split_folder']
        data_dir_map = args.paths['data_dir']
        train_csv = os.path.join(split_folder, f'fold_{args.fold}', 'train.csv')
        val_csv = os.path.join(split_folder, f'fold_{args.fold}', 'val.csv')
        test_csv = os.path.join(split_folder, f'fold_{args.fold}', 'test.csv')
        _, _, test_dataset = return_splits_tcga(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            test_csv_path=test_csv,
            data_dir_map=data_dir_map,
            label_dict=label_dict,
            seed=42,
            print_info=False
        )
        return test_dataset

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

def main(args):
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{args.fold}.csv')
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    pred_df = pd.read_csv(pred_path)
    print(f"[INFO] Loaded predictions for fold {args.fold}: {pred_df.shape[0]} samples")

    # Pre-filter by class
    class_dfs = {label: pred_df[pred_df['pred_label'] == label] for label in pred_df['pred_label'].unique()}
    class_counts = pred_df['pred_label'].value_counts().sort_index()
    print(f"[INFO] Class counts: {class_counts.to_dict()}")

    test_dataset = load_dataset(args)
    
    # Preload all slide features
    slide_ids = pred_df['slide_id'].unique()
    print(f"[INFO] Preloading features for {len(slide_ids)} slides")
    features_cache = preload_contrastive_features(dataset=test_dataset, slide_ids=slide_ids, max_workers=8)

    baseline_key = f'baseline_dir_fold_{args.fold}'
    save_dir = args.paths[baseline_key]
    os.makedirs(save_dir, exist_ok=True)

    # Collect baselines
    baselines = {}
    num_classes = len(args.label_dict)
    for target_class in range(num_classes):
        other_classes = [c for c in range(num_classes) if c != target_class]
        slide_subset =