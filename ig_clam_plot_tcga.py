import os
import sys 
import torch
import glob
import argparse
import h5py
import numpy as np
import openslide
import pandas as pd 

from utils_plot import (
    plot_heatmap_nobbox,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
)

def load_config(config_file):
    import yaml 
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def find_slide_path_mapping(basename, slide_root):
    pattern = os.path.join(slide_root, "*", f"{basename}.svs")
    paths = glob.glob(pattern)
    print("Searching:", pattern)
    print("Found paths:", paths)
    return paths[0] if paths else None

def plot_for_class(args, method, fold, class_id, score_dir, plot_dir):
    all_scores_paths = sorted(glob.glob(os.path.join(score_dir, "*.npy")))
    os.makedirs(plot_dir, exist_ok=True)

    already_plotted = {f.split(".")[0] for f in os.listdir(plot_dir) if f.endswith(".png")}
    
    scores_to_plot = [
        p for p in all_scores_paths 
        if os.path.splitext(os.path.basename(p))[0] not in already_plotted
    ]
      
    dataset_name = args.config_data["dataset_name"].lower()
    print(f"[Fold {fold} | Class {class_id}] Found {len(scores_to_plot)} new .npy files to plot")
    
    error_list = [] 
    for idx, score_path in enumerate(scores_to_plot):
        print(f"  → Plotting [{idx+1}/{len(scores_to_plot)}]: {score_path}")

        basename = os.path.splitext(os.path.basename(score_path))[0]

        if dataset_name == "camelyon16":
            slide_path = os.path.join(args.slide_path, f"{basename}.tif")

        elif dataset_name == "tcga_renal":
            label_dict = dict(config["label_dict"])
            id_to_label = {v: k for k, v in label_dict.items()}
            class_label_lower = id_to_label[class_id].lower()
            slide_root = args.slide_path_root[class_label_lower]
            slide_path = find_slide_path_mapping(basename, slide_root)

            if slide_path is None:
                error_list.append(basename) 
                print(f"  Slide for {basename} not found in {class_label_lower}, skipping.")
                 
        else:
            raise ValueError("Unknown dataset.")

        print("Slide path:", slide_path)
        if not os.path.exists(slide_path):
            print(f"  Slide not found: {slide_path}, skipping.")
            continue

        slide = openslide.open_slide(slide_path)
        _, new_width, new_height, original_width, original_height = rescaling_stat_for_segmentation(slide, downsampling_size=1096)
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        h5_path = os.path.join(args.features_h5_path, f"{basename}.h5")
        if not os.path.exists(h5_path):
            print(f"  H5 not found: {h5_path}, skipping.")
            continue

        with h5py.File(h5_path, "r") as f:
            coordinates = f['coords'][:]

        scores = np.load(score_path)
        clipped_scores = replace_outliers_with_bounds(scores.copy())
        scaled_scores = min_max_scale(clipped_scores)

        save_path = os.path.join(plot_dir, f"{basename}.png")
        plot_heatmap_nobbox(
            scale_x, scale_y, new_height, new_width,
            coordinates, scaled_scores, name="", save_path=save_path
        )
        print(f"  Saved to {save_path}")

    print(f"  Error list: {error_list}") 
    print(f"  Total errors: {len(error_list)}")
    print(f"  Length of scores to plot: {len(scores_to_plot)}")
    print(f"  Number of already plotted: {len(already_plotted)}") 

def main(args, config):
    dataset_name = config.get("dataset_name", "").lower()
    paths = config["paths"]

    args.slide_path = paths["slide_dir"] if dataset_name == "camelyon16" else None
    args.slide_path_root = paths["slide_dir"] if dataset_name == "tcga_renal" else None
    args.features_h5_path = paths["h5_files"]
    base_score_folder = paths["attribution_scores_folder"]
    base_plot_folder = paths["ig_clam_plot_folder"]
    args.config_data = config

    if dataset_name == "camelyon16":
        classes = [1, 0]
    elif dataset_name == "tcga_renal":
        classes = [0, 1, 2]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    for fold in range(args.start_fold, args.end_fold + 1):
        for class_id in classes:
            score_dir = os.path.join(
                base_score_folder, args.ig_name,
                f"fold_{fold}", f"class_{class_id}"
            )
            plot_dir = os.path.join(
                base_plot_folder, dataset_name, "plots_nobar",
                args.ig_name, f"fold_{fold}", f"class_{class_id}"
            )

            if not os.path.exists(score_dir):
                print(f"  Score folder not found: {score_dir}, skipping...")
                continue

            plot_for_class(args, args.ig_name, fold, class_id, score_dir, plot_dir)
            print("-------------")
        print("===================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--ig_name', required=True, help='Attribution method name')
    parser.add_argument('--start_fold', type=int, required=True)
    parser.add_argument('--end_fold', type=int, required=True)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config) 
    # main(args, config)
    
    missing_files = ['TCGA-A3-3313-01Z-00-DX1.10f8cd5e-1c4f-4bb4-96f3-c612f093fc80']
    slide_path = '/home/mvu9/datasets/TCGA-datasets/'
    for f in missing_files:
        print(f)
        path_pattern  = os.path.join(slide_path,"*/*", f"{f}.svs")
        files = glob.glob(path_pattern)
        print(files)
        
    score_path = '/home/mvu9/cig_attributions/attr_scores/tgca_renal'
    path_pattern = os.path.join(score_path,"*","f"))
    print(glob.glob(path_pattern))
