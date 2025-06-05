import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import openslide
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image

def main(args):
    fold_id = args.fold if hasattr(args, "fold") else 1
    meta_path = os.path.join(args.paths['metadata_plot_dir'], f"meta_fold_{fold_id}.pkl")
    plot_dir = os.path.join(args.paths['gt_plot'])

    print(f"[INFO] Loading metadata from: {meta_path}")
    meta_df = pd.read_pickle(meta_path)
    print(f"[INFO] Loaded {len(meta_df)} entries.")

    total_start = time.time()

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Saving downscaled slides"):
        slide_id = row['slide_id']
        slide_path = row['slide_path']
        new_width = int(row['new_width'])
        new_height = int(row['new_height'])
        save_path = os.path.join(plot_dir, f"{slide_id}.png")

        if not os.path.exists(slide_path):
            print(f"[WARN] Slide not found: {slide_path}")
            continue

        start_time = time.time()

        try:
            slide = openslide.OpenSlide(slide_path)
            slide_img = slide.read_region((0, 0), 0, slide.dimensions).convert("RGB")
            slide_img_resized = slide_img.resize((new_width, new_height), Image.BILINEAR)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            slide_img_resized.save(save_path)
            print(f"✅ Saved downscaled slide to {save_path} ({time.time() - start_time:.2f}s)")

        except Exception as e:
            print(f"[ERROR] Failed on {slide_id}: {e}")

    print(f"\n✅ All slides processed. Total time: {time.time() - total_start:.2f} seconds")


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

    main(args)
