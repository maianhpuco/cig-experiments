import os
import sys 
import torch
import glob
import argparse
import h5py
import numpy as np
import openslide

from utils_plot import (
    plot_heatmap_with_bboxes_nobar,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
)


def load_config(config_file):
    import yaml 
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def plot_for_class(args, method, fold, class_id, score_dir, plot_dir):
    all_scores_paths = glob.glob(os.path.join(score_dir, "*.npy"))
    os.makedirs(plot_dir, exist_ok=True)

    already_plotted = {f.split(".")[0] for f in os.listdir(plot_dir) if f.endswith(".png")}
    scores_to_plot = [p for p in all_scores_paths if os.path.basename(p).split(".")[0] not in already_plotted]

    print(f"[Fold {fold} | Class {class_id}] Found {len(scores_to_plot)} new .npy files to plot")

    for idx, score_path in enumerate(scores_to_plot):
        print(f"  → Plotting [{idx+1}/{len(scores_to_plot)}]: {score_path}")
        basename = os.path.basename(score_path).split(".")[0]
        slide_path = os.path.join(args.slide_path, f"{basename}.tif")

        if not os.path.exists(slide_path):
            print(f"  ⚠️  Slide not found: {slide_path}, skipping.")
            continue

        slide = openslide.open_slide(slide_path)

        (
            _, new_width, new_height,
            original_width, original_height
        ) = rescaling_stat_for_segmentation(slide, downsampling_size=1096)

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        h5_path = os.path.join(args.features_h5_path, f"{basename}.h5")
        if not os.path.exists(h5_path):
            print(f"  ⚠️  H5 not found: {h5_path}, skipping.")
            continue

        with h5py.File(h5_path, "r") as f:
            coordinates = f['coords'][:]

        scores = np.load(score_path)
        scaled_scores = min_max_scale(replace_outliers_with_bounds(scores.copy()))

        save_path = os.path.join(plot_dir, f"{basename}.png")
        plot_heatmap_with_bboxes_nobar(
            scale_x, scale_y, new_height, new_width,
            coordinates, scaled_scores, name="", save_path=save_path
        )
        print(f"  ✅ Saved to {save_path}")

def main(args, config):
    
    dataset_name = config.get("dataset_name", "").lower()
    paths = config["paths"]

    args.slide_path = paths["slide_dir"]
    args.features_h5_path = paths["h5_files"]
    base_score_folder = paths["attribution_scores_folder"]
    base_plot_folder = paths["ig_clam_plot_folder"]  # required key in config

    classes = []
    if dataset_name == "camelyon16":
        classes = [0, 1]
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
                print(f"⚠️  Score folder not found: {score_dir}, skipping...")
                continue
            plot_for_class(args, args.ig_name, fold, class_id, score_dir, plot_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--ig_name', required=True, help='Attribution method name')
    parser.add_argument('--start_fold', type=int, required=True)
    parser.add_argument('--end_fold', type=int, required=True)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config) 
    
    main(args, config)
