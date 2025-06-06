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
from utils_anno import extract_coordinates, check_xy_in_coordinates_fast

PATCH_SIZE = 256  # Match H5 patch size (try 224 if needed)

def read_h5_data(file_path, dataset_name="coords"):
    """Read dataset (default 'coords') from H5 file."""
    try:
        with h5py.File(file_path, "r") as file:
            if dataset_name in file:
                return file[dataset_name][()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None

# def reset_directory(path):
#     """Reset or create directory."""
#     if os.path.exists(path):
#         print(f"Resetting directory: {path}")
#         shutil.rmtree(path)
#     os.makedirs(path)

def main(args):
    """Process XML annotations and H5 patches to generate tumor masks."""
    # for key, path in args.paths.items():
    #     if not os.path.exists(path):
    #         print(f"[ERROR] Directory not found: {path} ({key})")
    #         return

    ann_list = [f for f in os.listdir(args.paths["annotation_dir"]) if f.endswith(".xml")]
    existing_csvs = set(os.listdir(args.paths["ground_truth_corr_dir"]))
    h5_files = set(os.listdir(args.paths["h5_files"]))
    
    valid_annotations = [
        f for f in ann_list
        if f.replace(".xml", ".h5") in h5_files and
        f.replace(".xml", ".csv") not in existing_csvs
    ]
    valid_annotations = ann_list 
    print(f"Total files to process: {len(valid_annotations)}")
    if not valid_annotations:
        print("[WARN] No new annotations to process")
        return

    processed = 0
    for idx, xml_file in enumerate(valid_annotations, 1):
        base = os.path.splitext(xml_file)[0]
        print(f"\n>>> [{idx}/{len(valid_annotations)}] Processing: {xml_file}")

        xml_path = os.path.join(args.paths["annotation_dir"], xml_file)
        h5_path = os.path.join(args.paths["h5_files"], f"{base}.h5")
        csv_save_path = os.path.join(args.paths["ground_truth_corr_dir"], f"{base}.csv")
        mask_save_path = os.path.join(args.paths["ground_truth_numpy_dir"], f"{base}.npy")

        # Read H5 coordinates
        coords = read_h5_data(h5_path)
        if coords is None or len(coords) == 0:
            print(f"[WARN] No coordinates in {h5_path}, skipping")
            continue

        # Extract coordinates
        df_xml = extract_coordinates(xml_path, csv_save_path, h5_coords=coords, patch_size=PATCH_SIZE)
        if df_xml is None or df_xml.empty:
            print(f"[WARN] No valid coordinates in {xml_file}, skipping")
            continue

        print(f"Saved coordinates to: {csv_save_path}, shape: {df_xml.shape}")

        # Generate mask
        mask = check_xy_in_coordinates_fast(df_xml, coords, patch_size=PATCH_SIZE)
        np.save(mask_save_path, mask)
        unique, counts = np.unique(mask, return_counts=True)
        label_counts = dict(zip(unique, counts))
        print(f"Saved mask to: {mask_save_path}, label counts: {label_counts}")
        print(f"Saved mask to: {mask_save_path}, sum={np.sum(mask)}")

        # Verify .pt file
        pt_path = os.path.join(args.paths["feature_files"], f"{base}.pt")
        if os.path.exists(pt_path):
            try:
                features = torch.load(pt_path, weights_only=True, map_location="cpu")
                print(f"✅ {base} | mask shape: {mask.shape}, pt shape: {features.shape}")
                if features.shape[0] != mask.shape[0]:
                    print(f"❌ Shape mismatch! Mask: {mask.shape[0]}, PT: {features.shape[0]}")
                else:
                    print(f"✔️ Patch count matches: {features.shape[0]} patches")
            except Exception as e:
                print(f"[ERROR] Failed to load {pt_path}: {e}")
        else:
            print(f"[WARN] PT file not found: {pt_path}")

        processed += 1
        # return 
    total_csv = len(os.listdir(args.paths["ground_truth_corr_dir"]))
    total_mask = len(os.listdir(args.paths["ground_truth_numpy_dir"]))
    print(f"\n✅ Finished.")
    print(f"+ Total annotations processed: {processed}")
    print(f"+ Total CSVs: {total_csv}")
    print(f"+ Total masks: {total_mask}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Camelyon16 annotations and patches")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config {args.config}: {e}")
        sys.exit(1)

    for key, val in config.items():
        setattr(args, key, val)

    for path in [args.paths["ground_truth_corr_dir"], args.paths["ground_truth_numpy_dir"]]:
        os.makedirs(path, exist_ok=True)

    main(args)