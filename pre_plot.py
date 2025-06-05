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
from utils_plot import (
    rescaling_stat_for_segmentation, 
)

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
    total = len(basenames)
    for idx, basename in enumerate(tqdm(basenames, desc="Processing slides")):
        dataset_name = args.dataset_name

        if dataset_name == "camelyon16":
            slide_path = os.path.join(args.paths['slide_root'], f"{basename}.tif")
            features_h5_dir = args.paths['h5_files']
            h5_path = glob(os.path.join(features_h5_dir, f"{basename}.h5"))  
        elif dataset_name in ["tcga_renal", 'tcga_lung']:
            slide_path = find_slide_path_mapping(basename, args.paths['slide_root'])
            features_h5_pattern = args.patterns['h5_files']
            h5_path = glob(os.path.join(features_h5_pattern, f"{basename}.h5"))  
        else:
            raise ValueError("Unknown dataset.")

        # try:
        print(f"Slide number {idx}/{total} - Reading slide at {slide_path}; h5_file at {h5_path}")
        slide = openslide.open_slide(slide_path)    

        _, new_width, new_height, original_width, original_height = rescaling_stat_for_segmentation(
            slide, downsampling_size=1096)
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        
        if len(h5_path) == 0 or not os.path.exists(h5_path[0]):
            print(f"[WARN] H5 not found for {basename}")
            error_list.append(basename)
            continue

        with h5py.File(h5_path[0], "r") as f:
            coords = f['coords'][:]
    
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
    os.makedirs(args.paths['metadata_plot_dir'], exist_ok=True)
    output_path = os.path.join(args.paths['metadata_plot_dir'], f"meta_fold_{fold_id}.pkl")
    meta_df.to_pickle(output_path)
    
    print(f"\n✅ Metadata saved to: {output_path}")



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

    extract_slide_metadata(args)
 
    extract_slide_metadata(args)

