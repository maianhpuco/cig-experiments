import os
import sys 
import torch
import glob
import argparse
import h5py
import numpy as np

def load_config(config_file):
    import yaml 
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def replace_outliers_with_bounds(array):
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    array = np.where(array < lower_bound, lower_bound, array)
    array = np.where(array > upper_bound, upper_bound, array)

    return array

def min_max_scale(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

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
        normalized_scores = min_max_scale(replace_outliers_with_bounds(raw_scores.copy()))        print(f"  📐 Shape         : {normalized_scores.shape}")
        
        print(f"  🔢 First 3 values: {[float(f'{s:.6f}') for s in normalized_scores[:3]]}")
        print(f"  🧮 Average       : {np.mean(normalized_scores):.6f}")
        print(f"  🔎 Non-zero count: {np.count_nonzero(normalized_scores)} / {len(normalized_scores)}")

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
