import os
import sys
import yaml
import argparse
import numpy as np
import openslide
import h5py
import pandas as pd
from glob import glob
from tqdm import tqdm

sys.path.extend([
    os.path.join("src/evaluation")
])

from utils.visualization_utils import rescaling_stat_for_segmentation


def find_slide_path_mapping(basename, slide_root):
    pattern = os.path.join(slide_root, "*/*", f"{basename}.svs")
    paths = glob(pattern)
    return paths[0] if paths else None


def extract_slide_metadata(args):
    fold_id = args.fold = 1
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    basenames = pred_df['slide_id'].unique().tolist()

    print(f"[INFO] Found {len(basenames)} slides for metadata extraction")

    meta_list = []
    error_list = []

    for idx, basename in enumerate(tqdm(basenames, desc="Processing slides")):
        dataset_name = args.dataset_name

        if dataset_name == "camelyon16":
            slide_path = os.path.join(args.slide_path, f"{basename}.tif")
        elif dataset_name == "tcga_renal":
            slide_path = find_slide_path_mapping(basename, args.slide_path)
            if slide_path is None:
                error_list.append(basename)
                continue
        else:
            raise ValueError("Unknown dataset.")

        try:
            slide = openslide.open_slide(slide_path)
        except Exception as e:
            print(f"[ERROR] Failed to open slide {basename}: {e}")
            error_list.append(basename)
            continue

        try:
            _, new_width, new_height, original_width, original_height = rescaling_stat_for_segmentation(slide, downsampling_size=1096)
            scale_x = new_width / original_width
            scale_y = new_height / original_height
        except Exception as e:
            print(f"[ERROR] Failed to get rescaling stats for {basename}: {e}")
            error_list.append(basename)
            continue

        h5_path = glob(os.path.join(args.features_h5_pattern, f"{basename}.h5"))
        if len(h5_path) == 0 or not os.path.exists(h5_path[0]):
            print(f"[WARN] H5 not found for {basename}")
            error_list.append(basename)
            continue

        try:
            with h5py.File(h5_path[0], "r") as f:
                coords = f['coords'][:]
        except Exception as e:
            print(f"[ERROR] Failed to read h5 coords for {basename}: {e}")
            error_list.append(basename)
            continue

        meta_list.append({
            "slide_id": basename,
            "slide_path": slide_path,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "new_width": new_width,
            "new_height": new_height,
            "coords": coords.tolist(),  # store as list to make JSON/CSV saveable
        })

    meta_df = pd.DataFrame(meta_list)
    os.makedirs(args.paths['metadata_dir'], exist_ok=True)
    output_path = os.path.join(args.paths['metadata_dir'], f"meta_fold_{fold_id}.pkl")
    meta_df.to_pickle(output_path)
    print(f"\n✅ Metadata saved to: {output_path}")

    if error_list:
        print("\n⚠️ Some slides failed:")
        for e in error_list:
            print(" -", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ig_name', type=str, required=True)
    
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

    extract_slide_metadata(args)

