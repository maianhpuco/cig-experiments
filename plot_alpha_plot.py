import os
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

def plot_heatmap_nobbox(scale_x, scale_y, new_height, new_width, coordinates, scores,
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
        color = cmap(norm(scores[i]))
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


def plot_all_intermediate_alpha(args):
    fold_id = args.fold if hasattr(args, "fold") else 1
    meta_path = os.path.join(args.paths['metadata_plot_dir'], f"meta_fold_{fold_id}.pkl")
    score_dir = os.path.join(args.paths['attr_score_for_multi_alpha_plot_dir'], args.ig_name, f"fold_{fold_id}")
    plot_dir = os.path.join(args.paths['multi_alpha_plot_dir'], args.ig_name, f"fold_{fold_id}")

    print(f"[INFO] Loading metadata from: {meta_path}")
    meta_df = pd.read_pickle(meta_path)
    print(f"[INFO] Loaded {len(meta_df)} entries.")

    total_start = time.time()
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Plotting all alpha steps"):
        slide_id = row['slide_id']
        score_path = os.path.join(score_dir, slide_id, "attr_alpha_avg.npy")

        if not os.path.exists(score_path):
            print(f"[WARN] Score not found: {score_path}")
            continue

        alpha_values = np.load(score_path)  # shape [7, N]
        if alpha_values.ndim != 2 or alpha_values.shape[0] != 7:
            print(f"[ERROR] Invalid shape {alpha_values.shape} in {score_path}")
            continue

        for i in range(7):
            alpha_score = alpha_values[i]  # [N]

            # Normalize per alpha
            clipped = np.clip(alpha_score, np.percentile(alpha_score, 1), np.percentile(alpha_score, 99))
            scaled = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)

            alpha_plot_dir = os.path.join(plot_dir, f"alpha_{i+1}")
            save_path = os.path.join(alpha_plot_dir, f"{slide_id}.png")

            plot_heatmap_nobbox(
                scale_x=row['scale_x'],
                scale_y=row['scale_y'],
                new_height=row['new_height'],
                new_width=row['new_width'],
                coordinates=row['coords'],
                scores=scaled,
                save_path=save_path,
                name=f"{slide_id} | α-{i+1}"
            )

    print(f"\n✅ All alpha plots done. Total time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--ig_name', required=True, help='Attribution method name (e.g., cig, ig)')
    parser.add_argument('--fold', type=int, default=1, help='Fold ID')
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    plot_all_intermediate_alpha(args)
