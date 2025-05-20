import os
import sys 
import torch
import glob
import argparse
import h5py
import numpy as np

# from utils_plot import (
#     min_max_scale, 
#     replace_outliers_with_bounds
# )

def min_max_scale(array):
    """
    Normalize a numpy array to the [0, 1] range after min-max scaling.

    If all elements are equal, returns a zero array of the same shape.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(array)
    
    return (array - min_val) / range_val
 
def load_config(config_file):
    import yaml 
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def inspect_scores_for_class(args, method, fold, class_id, score_dir):
    all_scores_paths = sorted(glob.glob(os.path.join(score_dir, "*.npy")))[:3]  # only first 3
    print(f"[Fold {fold} | Class {class_id}] Inspecting {len(all_scores_paths)} score files")

    for idx, score_path in enumerate(all_scores_paths):
        basename = os.path.basename(score_path).split(".")[0]
        print(f"\n🔍 [{idx+1}/3] Inspecting: {basename}")

        if not os.path.exists(score_path):
            print(f"  ⚠️  Score file not found: {score_path}, skipping.")
            continue

        raw_scores = np.load(score_path)
        normalized_scores = raw_scores
        # normalized_scores = min_max_scale(raw_scores.copy())
        # normalized_scores = raw_scores 
        print(f"  >  Shape          : {normalized_scores.shape}")
        print(f"  >  First 3 values : {[float(f'{s:.6f}') for s in normalized_scores[:3]]}")
        print(f"  >  Sum            : {np.sum(normalized_scores):.6f}")
        print(f"  >  Min value      : {np.min(normalized_scores):.6f}")
        print(f"  >  Max value      : {np.max(normalized_scores):.6f}")
        print(f"  >  Non-zero count : {np.count_nonzero(normalized_scores)} / {len(normalized_scores)}")
 
def main(args, config):
    dataset_name = config.get("dataset_name", "").lower()
    paths = config["paths"]

    base_score_folder = paths["attribution_scores_folder"]

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
            if not os.path.exists(score_dir):
                print(f"\n⚠️  Score folder not found: {score_dir}, skipping...")
                continue

            inspect_scores_for_class(args, args.ig_name, fold, class_id, score_dir)
            print("-------")
        print("---------------")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    
    args = parser.parse_args()

    # Defaults
    args.start_fold = 1
    args.end_fold = 1
    args.ig_name = "contrastive_gradient" 
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = load_config(args.config)
    main(args, config)
