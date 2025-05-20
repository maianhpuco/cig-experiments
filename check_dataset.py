import os
import sys
import argparse
import yaml
import torch
from src.datasets.classification.tcga import return_splits_custom  
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon

def load_dataset(args):
    if args.dataset_name == 'tcga_renal':
        split_folder = args.paths['split_folder']
        data_dir_map = args.paths['data_dir']
        label_dict = {'KICH': 0, 'KIRP': 1, 'KIRC': 2}
        fold_id = 1

        train_csv = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
        val_csv = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
        test_csv = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

        train_ds, val_ds, test_ds = return_splits_custom(
            train_csv, val_csv, test_csv,
            data_dir_map=data_dir_map,
            label_dict=label_dict,
            seed=42,
            print_info=False
        )
        return test_ds

    elif args.dataset_name == 'camelyon16':
        fold_id = 1
        csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}

        train_ds, _, test_ds = return_splits_camelyon(
            csv_path=csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=label_dict,
            seed=args.seed,
            print_info=False,
            use_h5=True
        )
        return test_ds

def main(args):
    test_dataset = load_dataset(args)
    print(f"-- Loaded test dataset with {len(test_dataset)} samples --")

    for idx, (features, label, coords) in enumerate(test_dataset):
        print(f"\nSample {idx + 1}")
        print(f" - Feature shape: {features.shape}")
        print(f" - Label: {label}")
        print(f" - Coords shape: {coords.shape}")
        if idx >= 2:  # just print 3 samples
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
