import os
import sys
import yaml
import argparse
import numpy as np
import openslide
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from glob import glob
from tqdm import tqdm
import pandas as pd 

sys.path.extend([
    os.path.join("src/models"),
    os.path.join("src/evaluation")
])

from utils.visualization_utils import replace_outliers_with_bounds, min_max_scale, rescaling_stat_for_segmentation
from clam import load_clam_model


def find_slide_path_mapping(basename, slide_root):
    pattern = os.path.join(slide_root, "*/*", f"{basename}.svs")
    paths = glob(pattern)
    return paths[0] if paths else None


def plot_heatmap_nobbox(scale_x, scale_y, new_height, new_width, coordinates, scores, figsize=(10, 10), name="", save_path=None, patch_size=256):
    cmap = cm.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    fig, ax = plt.subplots(figsize=figsize)
    white_background = np.ones((new_height, new_width, 3))
    ax.imshow(white_background)
    ax.axis('off')

    for i, coord in tqdm(enumerate(coordinates), total=len(coordinates), desc="Plotting heatmap patches"):
        x, y = coord.astype('int')
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        scaled_patch_w = patch_size * scale_x
        scaled_patch_h = patch_size * scale_y
        color = cmap(scores[i])

        rect = patches.Rectangle((scaled_x, scaled_y), scaled_patch_w, scaled_patch_h,
                                 linewidth=0.0, edgecolor=color, facecolor=color)
        ax.add_patch(rect)

    plt.title(name, fontsize=10, fontweight='bold')
    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved heatmap to {save_path}")
    plt.close(fig)



def main(args):
    fold_id = args.fold = 1
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)

    if args.dataset_name == "camelyon16":
        pred_df = pred_df[pred_df['pred_label'] == 1]
        print(f"[INFO] Camelyon16: filtering to pred_label == 1 --> {len(pred_df)} slides")

    basenames = pred_df['slide_id'].unique().tolist()

    score_dir = os.path.join(args.paths['attribution_scores_folder'], args.ig_name, f"fold_{fold_id}")
    plot_dir = os.path.join(args.paths['attribution_scores_plot'], args.ig_name, f"fold_{fold_id}")

    print(f"[INFO] Loaded {len(basenames)} slides for plotting")
    error_list = []

    for idx, basename in enumerate(basenames):
        print(f"\n=== Plotting slide: {basename} ({idx + 1}/{len(basenames)}) ===")
        dataset_name = args.dataset_name

        # Slide path resolution
        if dataset_name == "camelyon16":
            slide_path = os.path.join(args.slide_path, f"{basename}.tif")
        elif dataset_name == "tcga_renal":
            slide_path = find_slide_path_mapping(basename, args.slide_path)
            if slide_path is None:
                error_list.append(basename)
                print(f"  Slide for {basename} not found, skipping.")
                continue
        else:
            raise ValueError("Unknown dataset.")

        try:
            slide = openslide.open_slide(slide_path)
        except Exception as e:
            print(f"  Failed to open slide {basename}: {e}")
            error_list.append(basename)
            continue

        _, new_width, new_height, original_width, original_height = rescaling_stat_for_segmentation(slide, downsampling_size=1096)
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        h5_path = glob(os.path.join(args.features_h5_pattern, f"{basename}.h5"))
        if len(h5_path) == 0 or not os.path.exists(h5_path[0]):
            print(f"  H5 not found: {h5_path}, skipping.")
            error_list.append(basename)
            continue

        with h5py.File(h5_path[0], "r") as f:
            coordinates = f['coords'][:]

        score_path = os.path.join(score_dir, f"{basename}.npy")
        if not os.path.exists(score_path):
            print(f"  Score not found: {score_path}, skipping.")
            error_list.append(basename)
            continue

        scores = np.load(score_path)
        clipped_scores = replace_outliers_with_bounds(scores.copy())
        scaled_scores = min_max_scale(clipped_scores)

        save_path = os.path.join(plot_dir, f"{basename}.png")
        plot_heatmap_nobbox(
            scale_x, scale_y, new_height, new_width,
            coordinates, scaled_scores, name="", save_path=save_path
        )

    if error_list:
        print("\n⚠️ The following slides could not be processed:")
        for e in error_list:
            print(" -", e)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ig_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
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

    main(args)

