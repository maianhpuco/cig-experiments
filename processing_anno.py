import sys
import argparse
import torch
import h5py
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import os 
import yaml 
from utils_anno import (
    extract_coordinates,
    check_xy_in_coordinates_fast,
)

def read_h5_data(file_path, dataset_name=None):
    with h5py.File(file_path, "r") as file:
        if dataset_name:
            if dataset_name in file:
                return file[dataset_name][()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
        else:
            datasets = {name: node[()] for name, node in file.items() if isinstance(node, h5py.Dataset)}
            return next(iter(datasets.values())) if len(datasets) == 1 else datasets

def reset_directory(path):
    if os.path.exists(path):
        print(f"Resetting directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path)

def main(args):
    ann_list = os.listdir(args.paths['annotation_dir'])
    existing_csvs = os.listdir(args.paths['ground_truth_corr_dir'])
    h5_files = os.listdir(args.paths['h5_files'])

    valid_annotations = [f for f in ann_list if f.endswith(".xml") and f.replace(".xml", ".csv") not in existing_csvs and f.replace(".xml", ".h5") in h5_files]

    print(f"Total files to process: {len(valid_annotations)}")

    for idx, xml_file in enumerate(valid_annotations):
        base = os.path.splitext(xml_file)[0]
        print(f"\\n>>> [{idx+1}/{len(valid_annotations)}] Processing: {xml_file}")

        h5_path = os.path.join(args.paths['h5_files'], f"{base}.h5")
        xml_path = os.path.join(args.paths['annotation_dir'], xml_file)
        csv_save_path = os.path.join(args.paths['ground_truth_corr_dir'], f"{base}.csv")

        df_xml = extract_coordinates(xml_path, csv_save_path)
        if df_xml is None:
            print(f"[WARN] No valid contour in {xml_file}, skipping.")
            continue

        print(f"Saved contour to: {csv_save_path}, shape: {df_xml.shape}")

        h5_data = read_h5_data(h5_path)
        coords = h5_data["coordinates"] if isinstance(h5_data, dict) else h5_data

        mask = check_xy_in_coordinates_fast(df_xml, coords)
        mask_save_path = os.path.join(args.paths['ground_truth_numpy_dir'], f"{base}.npy")
        np.save(mask_save_path, mask)
        print(f"Saved mask to: {mask_save_path}, sum={np.sum(mask)}")
        # Load corresponding .pt file
        
        pt_path = os.path.join(args.paths['feature_files'], f"{base}.pt")
        if os.path.exists(pt_path):
            features = torch.load(pt_path)
            print(f"✅ {base} | mask shape: {mask.shape}, pt shape: {features.shape}")

            if features.shape[0] != mask.shape[0]:
                print(f"❌ Shape mismatch! Mask: {mask.shape[0]}, PT: {features.shape[0]}")
            else:
                print(f"✔️  Patch count matches: {features.shape[0]} patches")
        else:
            print(f"[WARN] PT file not found: {pt_path}") 
    total_csv = len(os.listdir(args.paths['ground_truth_corr_dir']))
    total_mask = len(os.listdir(args.paths['ground_truth_numpy_dir']))
    print(f"\n ✅ Finished.")
    print(f"+ Total annotations processed: {len(valid_annotations)}")
    print(f"+ Total CSVs: {total_csv}")
    print(f"+ Total masks: {total_mask}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    # args.paths['annotation_dir'] = r"{annotation_dir}"
    # args.paths['h5_files'] = r"{h5_files}"
    # args.paths['ground_truth_corr_dir'] = r"{ground_truth_corr_dir}"
    # args.paths['ground_truth_numpy_dir'] = r"{ground_truth_numpy_dir}"

    for path in [args.paths['ground_truth_corr_dir'], args.paths['ground_truth_numpy_dir']]:
        os.makedirs(path, exist_ok=True)

    main(args)