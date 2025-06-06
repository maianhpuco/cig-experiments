import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from tqdm import tqdm
import time


def plot_mask_nobbox(scale_x, scale_y, new_height, new_width, coordinates, scores,
                     figsize=(10, 10), name="", save_path=None, patch_size=256):
    cmap = plt.colormaps['coolwarm']
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    fig, ax = plt.subplots(figsize=figsize)
    white_background = np.ones((new_height, new_width, 3))
    ax.imshow(white_background)
    ax.axis('off')

    for i, coord in enumerate(coordinates):
        x, y = np.array(coord).astype('int')
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved mask heatmap to {save_path}")
    plt.close(fig)


def plot_all_masks(meta_df, mask_dir, save_dir, patch_size=256):
    print(f"[INFO] Plotting masks from: {mask_dir}")
    total_start = time.time()

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Plotting masks"):
        slide_id = row['slide_id']
        mask_path = os.path.join(mask_dir, f"{slide_id}.npy")
        save_path = os.path.join(save_dir, f"{slide_id}.png")

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found: {mask_path}")
            continue

        mask_values = np.load(mask_path).squeeze()
        if mask_values.ndim != 1 or len(mask_values) != len(row['coords']):
            print(f"[ERROR] Shape mismatch in {slide_id}: mask shape = {mask_values.shape}, coords = {len(row['coords'])}")
            continue

        # Normalize for consistent colormap display
        clipped = np.clip(mask_values, np.percentile(mask_values, 1), np.percentile(mask_values, 99))
        scaled = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)

        plot_mask_nobbox(
            scale_x=row['scale_x'],
            scale_y=row['scale_y'],
            new_height=row['new_height'],
            new_width=row['new_width'],
            coordinates=row['coords'],
            scores=scaled,
            save_path=save_path,
            patch_size=patch_size,
            name=slide_id
        )

    print(f"\n✅ Finished plotting all masks. Total time: {time.time() - total_start:.2f} seconds")


def main(args):
    fold_id = getattr(args, "fold", 1)

    meta_path = os.path.join(args.paths['metadata_plot_dir'], f"meta_fold_{fold_id}.pkl")
    mask_dir = args.paths['ground_truth_numpy_dir']
    save_dir = os.path.join(args.paths['gt_heatmap_plot'])

    print(f"[INFO] Loading metadata from: {meta_path}")
    meta_df = pd.read_pickle(meta_path)
    print(f"[INFO] Loaded {len(meta_df)} entries.")

    plot_all_masks(meta_df, mask_dir, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tumor masks as heatmaps.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--fold', type=int, default=1, help='Fold number (default=1)')
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args_cmd.paths = val
        else:
            setattr(args_cmd, key, val)

    main(args_cmd)
