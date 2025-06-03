import os
import torch
import argparse
import yaml
from src.datasets.classification.camelyon16 import return_splits_custom


def compute_mean_variance(dataset):
    all_feats = []

    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict):
            feats = item['features']
        elif isinstance(item, tuple) or isinstance(item, list):
            feats = item[0]  # (features, label)
        else:
            feats = item
        feats = feats.view(-1, feats.shape[-1])  # [N, D]
        all_feats.append(feats)

    all_feats = torch.cat(all_feats, dim=0)  # [Total_N, D]
    mean = torch.mean(all_feats, dim=0)
    std = torch.std(all_feats, dim=0)
    return mean, std

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    split_folder = cfg['paths']['split_folder']
    data_dir = cfg['paths']['pt_files']
    output_dir = cfg['paths']['dataset_mean']  # Output directory from config
    os.makedirs(output_dir, exist_ok=True)

    label_dict = {'normal': 0, 'tumor': 1}

    for fold in range(args.start_fold, args.end_fold + 1):
        print(f"\n[Fold {fold}] Computing mean and std...")
        csv_path = os.path.join(split_folder, f'fold_{fold}.csv')

        train_dataset, _, _ = return_splits_custom(
            csv_path=csv_path,
            data_dir=data_dir,
            label_dict=label_dict,
            seed=42,
            print_info=False
        )

        mean, std = compute_mean_variance(train_dataset)

        output_path = os.path.join(output_dir, f'fold_{fold}_mean_std.pt')
        torch.save({'mean': mean, 'std': std}, output_path)
        print(f"[✓] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Fold-wise Mean and Std")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--start_fold', type=int, required=True, help='Start fold index (inclusive)')
    parser.add_argument('--end_fold', type=int, required=True, help='End fold index (inclusive)')
    args = parser.parse_args()
    main(args)
