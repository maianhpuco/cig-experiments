import os
import yaml
import argparse
import torch
import pandas as pd

from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16

def sample_contrastive_features(pred_df, dataset, target_slide_id, target_label, sample_classes, max_slides=5):
    target_feats = dataset.get_features_by_slide_id(target_slide_id)
    num_target_feats = target_feats.shape[0]
    print(f"[INFO] Target slide {target_slide_id} has {num_target_feats} features")

    sample_df = pred_df[pred_df['pred_label'].isin(sample_classes)].sample(frac=1, random_state=42)

    if len(sample_df) == 0:
        raise ValueError("No contrastive samples found.")

    sampled_features = []
    sampled_slides = sample_df.head(max_slides)

    num_slides = len(sampled_slides)
    feats_per_slide = num_target_feats // num_slides
    remainder = num_target_feats % num_slides

    current_count = 0
    for i, (_, row) in enumerate(sampled_slides.iterrows()):
        slide_id = row['slide_id']
        try:
            feats = dataset.get_features_by_slide_id(slide_id)
        except Exception as e:
            print(f"[WARNING] Could not load slide {slide_id}: {e}")
            continue

        feats = feats[torch.randperm(feats.shape[0])]

        take_n = feats_per_slide + (1 if i < remainder else 0)
        take_n = min(take_n, feats.shape[0])
        sampled_features.append(feats[:take_n])
        current_count += take_n

        print(f"[INFO] Sampled {take_n} from {slide_id}, running total: {current_count}")

    if len(sampled_features) == 0:
        raise RuntimeError("No contrastive features collected.")

    sampled_feats = torch.cat(sampled_features, dim=0)
    print(f"[INFO] Final shape: {sampled_feats.shape}")
    return sampled_feats
 

def main(args):
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{args.fold}.csv')
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    print(f"[INFO] Loaded predictions from fold {args.fold}: {pred_df.shape[0]} samples")

    train_csv = os.path.join(args.paths['split_folder'], f'fold_{args.fold}', 'train.csv')
    val_csv = os.path.join(args.paths['split_folder'], f'fold_{args.fold}', 'val.csv')
    test_csv = os.path.join(args.paths['split_folder'], f'fold_{args.fold}', 'test.csv')

    _, _, test_dataset = return_splits_custom(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        test_csv_path=test_csv,
        data_dir_map=args.paths['data_dir'],
        label_dict=args.label_dict,
        print_info=True,
        use_h5=args.use_h5
    )

    save_dir = args.paths[f'baseline_dir_fold_{args.fold}']
    os.makedirs(save_dir, exist_ok=True)

    num_classes = len(args.label_dict)
    for target_class in range(num_classes):
        other_classes = [c for c in range(num_classes) if c != target_class]
        slide_subset = pred_df[pred_df['pred_label'] == target_class]

        for _, row in slide_subset.iterrows():
            slide_id = row['slide_id']
            print(f"[INFO] Sampling contrastive features for slide {slide_id} (class {target_class})")

            sampled_feats = sample_contrastive_features(
                pred_df=pred_df,
                dataset=test_dataset,
                target_slide_id=slide_id,
                target_label=target_class,
                sample_classes=other_classes
            )

            save_path = os.path.join(save_dir, f"{slide_id}.pt")
            torch.save(sampled_feats, save_path)
            print(f"[INFO] Saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='configs_simea/clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--start_fold', type=int, default=1)
    parser.add_argument('--end_fold', type=int, default=1)

    args = parser.parse_args()

    with open(f'{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    print(" > Start computing contrastive samples for dataset:", args.dataset_name)
    for fold_id in range(args.start_fold, args.end_fold + 1):
        args.fold = fold_id
        main(args)
