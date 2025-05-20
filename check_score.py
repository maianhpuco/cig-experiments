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

def inspect_scores_for_class(args, method, fold, class_id, score_dir):
    all_scores_paths = glob.glob(os.path.join(score_dir, "*.npy"))
    print(f"[Fold {fold} | Class {class_id}] Found {len(all_scores_paths)} .npy score files")

    for idx, score_path in enumerate(all_scores_paths):
        basename = os.path.basename(score_path).split(".")[0]
        print(f"\n🔍 [{idx+1}/{len(all_scores_paths)}] Inspecting: {basename}")

        if not os.path.exists(score_path):
            print(f"  ⚠️  Score file not found: {score_path}, skipping.")
            continue

        scores = np.load(score_path)

        print(f"  📐 Shape         : {scores.shape}")
        print(f"  🔢 First 3 values: {[float(f'{s:.6f}') for s in scores[:3]]}")
        print(f"  🧮 Average       : {np.mean(scores):.6f}")
        print(f"  🔎 Non-zero count: {np.count_nonzero(scores)} / {len(scores)}")

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
    # parser.add_argument('--ig_name', required=True, help='Attribution method name')
    # parser.add_argument('--start_fold', type=int, required=True)
    # parser.add_argument('--end_fold', type=int, required=True)
    
    args = parser.parse_args()

    args.start_fold = 1
    args.end_fold = 1
    args.ig_name = "contrastive_gradient" 
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    main(args, config)
 
 
# python check_score.py --config configs_simea/clam_camelyon16.yaml