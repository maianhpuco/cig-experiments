import os
import yaml
import argparse
import torch
import pandas as pd

def sample_contrastive_features(pred_df, dataset, target_slide_id, target_label, sample_classes, max_slides=30):
    target_feats = dataset.get_features_by_slide_id(target_slide_id)
    num_target_feats = target_feats.shape[0]

    sample_df = pred_df[pred_df['pred_label'].isin(sample_classes)].sample(frac=1, random_state=42)
    if len(sample_df) == 0:
        raise ValueError("No contrastive samples found.")

    sampled_features = []
    sampled_slides = sample_df.head(max_slides)
    num_slides = len(sampled_slides)
    feats_per_slide = num_target_feats // num_slides
    remainder = num_target_feats % num_slides

    for i, (_, row) in enumerate(sampled_slides.iterrows()):
        slide_id = row['slide_id']
        try:
            feats = dataset.get_features_by_slide_id(slide_id)
        except Exception as e:
            print(f"[WARNING] Could not load slide {slide_id}: {e}")
            continue
        feats = feats[torch.randperm(feats.shape[0])]
        take_n = min(feats_per_slide + (1 if i < remainder else 0), feats.shape[0])
        sampled_features.append(feats[:take_n])
        print(f"[INFO] Contrastive slide: {slide_id} | Sampled: {take_n}")

    if len(sampled_features) == 0:
        raise RuntimeError("No contrastive features collected.")

    return torch.cat(sampled_features, dim=0)

def load_dataset(args, fold_id):
    if args.dataset_name == 'camelyon16':
        from src.datasets.classification.camelyon16 import return_splits_custom
        split_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        args.label_dict = {'normal': 0, 'tumor': 1}
        _, _, test_dataset = return_splits_custom(
            csv_path=split_csv,
            data_dir=args.paths['pt_files'],
            label_dict=args.label_dict,
            seed=42,
            print_info=True
        )
        return test_dataset

    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        from src.datasets.classification.tcga import return_splits_custom
        label_dict = getattr(args, "label_dict", None)
        fold_dir = os.path.join(args.paths['split_folder'], f'fold_{fold_id}')
        _, _, test_dataset = return_splits_custom(
            train_csv_path=os.path.join(fold_dir, 'train.csv'),
            val_csv_path=os.path.join(fold_dir, 'val.csv'),
            test_csv_path=os.path.join(fold_dir, 'test.csv'),
            data_dir_map=args.paths['data_dir'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        return test_dataset

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

def main(args):
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{args.fold}.csv')
    pred_df = pd.read_csv(pred_path)
    print(f"[INFO] Loaded predictions: {len(pred_df)} samples")

    test_dataset = load_dataset(args, args.fold)

    save_dir = args.paths.get(f'baseline_dir_fold_{args.fold}')
    if save_dir is None:
        raise KeyError(f"baseline_dir_fold_{args.fold} not found in paths.")
    os.makedirs(save_dir, exist_ok=True)

    for target_class in range(len(args.label_dict)):
        contrastive_classes = [c for c in range(len(args.label_dict)) if c != target_class]
        slides = pred_df[pred_df['pred_label'] == target_class]

        print(f"\n[INFO] Target class {target_class} — {len(slides)} slides")

        for idx, (_, row) in enumerate(slides.iterrows(), 1):
            slide_id = row['slide_id']
            print(f"[{idx}/{len(slides)}] Slide: {slide_id}")
            sampled_feats = sample_contrastive_features(
                pred_df, test_dataset, slide_id, target_class, contrastive_classes
            )
            save_path = os.path.join(save_dir, f"{slide_id}.pt")
            torch.save(sampled_feats, save_path)
            print(f"[INFO] Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--start_fold', type=int, default=1)
    parser.add_argument('--end_fold', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(args.start_fold, args.end_fold + 1):
        args.fold = fold
        print(f"\n=== Fold {fold} ===")
        main(args)
