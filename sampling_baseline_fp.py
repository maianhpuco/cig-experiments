import os
import yaml
import argparse
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def sample_contrastive_features(slide_id, target_label, sample_classes, pred_df, dataset, max_slides, feature_cache):
    target_feats = feature_cache[slide_id]
    num_target_feats = target_feats.shape[0]

    sample_df = pred_df[pred_df['pred_label'].isin(sample_classes)].sample(frac=1, random_state=42)
    if len(sample_df) == 0:
        raise ValueError("No contrastive samples found.")

    sampled_features = []
    sampled_slides = sample_df.head(max_slides)
    num_slides = len(sampled_slides)
    feats_per_slide = num_target_feats // num_slides
    remainder = num_target_feats % num_slides

    for i, row in enumerate(sampled_slides.itertuples(index=False)):
        contrast_slide_id = row.slide_id
        feats = feature_cache.get(contrast_slide_id)
        if feats is None:
            continue

        feats = feats[torch.randperm(feats.shape[0])]
        take_n = feats_per_slide + (1 if i < remainder else 0)
        sampled_features.append(feats[:take_n])

    if not sampled_features:
        raise RuntimeError("No contrastive features collected.")

    return torch.cat(sampled_features, dim=0)

def load_dataset(args, fold_id):
    if args.dataset_name == 'camelyon16':
        from src.datasets.classification.camelyon16 import return_splits_custom
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        args.label_dict = {'normal': 0, 'tumor': 1}
        _, _, test_dataset = return_splits_custom(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=args.label_dict,
            seed=42,
            print_info=True
        )
    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        from src.datasets.classification.tcga import return_splits_custom
        split_folder = args.paths['split_folder']
        data_dir_map = args.paths['data_dir']
        train_csv = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
        val_csv = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
        test_csv = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')
        _, _, test_dataset = return_splits_custom(
            train_csv, val_csv, test_csv,
            data_dir_map=data_dir_map,
            label_dict=args.label_dict,
            seed=42,
            print_info=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    return test_dataset

def process_slide(slide_id, target_class, other_classes, pred_df, dataset, save_dir, feature_cache):
    try:
        sampled_feats = sample_contrastive_features(
            slide_id, target_class, other_classes,
            pred_df, dataset, max_slides=30,
            feature_cache=feature_cache
        )
        save_path = os.path.join(save_dir, f"{slide_id}.pt")
        torch.save(sampled_feats, save_path)
        print(f"[SAVED] {slide_id} -> {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed for {slide_id}: {e}")

def main(args):
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{args.fold}.csv')
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    pred_df = pd.read_csv(pred_path)

    test_dataset = load_dataset(args, args.fold)
    baseline_key = f'baseline_dir_fold_{args.fold}'
    save_dir = args.paths.get(baseline_key)
    os.makedirs(save_dir, exist_ok=True)

    feature_cache = {ds.slide_ids[i]: ds.get_features_by_slide_id(ds.slide_ids[i]) for i, ds in enumerate(test_dataset)}
    num_classes = len(args.label_dict)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for target_class in range(num_classes):
            other_classes = [c for c in range(num_classes) if c != target_class]
            slide_subset = pred_df[pred_df['pred_label'] == target_class]
            print(f"\n[CLASS {target_class}] {len(slide_subset)} slides")
            for row in tqdm(slide_subset.itertuples(index=False), total=len(slide_subset)):
                executor.submit(
                    process_slide,
                    row.slide_id, target_class, other_classes,
                    pred_df, test_dataset, save_dir, feature_cache
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='configs_simea/clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--start_fold', type=int, default=1)
    parser.add_argument('--end_fold', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(" > Start computing contrastive samples for dataset:", args.dataset_name)
    for fold_id in range(args.start_fold, args.end_fold + 1):
        args.fold = fold_id
        main(args)
