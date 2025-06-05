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

def plot_heatmap_nobbox(scale_x, scale_y, new_height, new_width, coordinates, scores, figsize=(10, 10), name="", save_path=None, patch_size=256):
    cmap = cm.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    fig, ax = plt.subplots(figsize=figsize)
    white_background = np.ones((new_height, new_width, 3))
    ax.imshow(white_background)
    ax.axis('off')

    for i, coord in tqdm(enumerate(coordinates), total=len(coordinates), desc="Plotting heatmap patches"):
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
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved heatmap to {save_path}")
    plt.close(fig)

def main(args):
    fold_id = args.fold if hasattr(args, "fold") else 1

    meta_path = os.path.join(args.paths['metadata_plot_dir'], f"meta_fold_{fold_id}.pkl")
    score_dir = os.path.join(args.paths['attribution_scores_folder'], args.ig_name, f"fold_{fold_id}")
    plot_dir = os.path.join(args.paths['plot_folder'], args.ig_name, f"fold_{fold_id}")

    print(f"[INFO] Loading metadata from: {meta_path}")
    meta_df = pd.read_pickle(meta_path)
    print(f"[INFO] Loaded {len(meta_df)} entries.")

    for i, row in meta_df.iterrows():
        slide_id = row['slide_id']
        save_path = os.path.join(plot_dir, f"{slide_id}.png")
        score_path = os.path.join(score_dir, f"{slide_id}.npy")

        if not os.path.exists(score_path):
            print(f"[WARN] Score not found: {score_path}")
            continue

        scores = np.load(score_path)
        clipped_scores = np.clip(scores, np.percentile(scores, 1), np.percentile(scores, 99))
        scaled_scores = (clipped_scores - clipped_scores.min()) / (clipped_scores.max() - clipped_scores.min() + 1e-8)

        plot_heatmap_nobbox(
            scale_x=row['scale_x'],
            scale_y=row['scale_y'],
            new_height=row['new_height'],
            new_width=row['new_width'],
            coordinates=row['coords'],
            scores=scaled_scores,
            save_path=save_path
        )

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

    main(args)
