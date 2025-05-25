import os
import argparse
import numpy as np
import pandas as pd
import cv2
import yaml
from glob import glob

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def min_max_scale(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return np.zeros_like(array) if max_val == min_val else (array - min_val) / (max_val - min_val)

def replace_outliers_with_bounds(array):
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    array = np.clip(array, lower_bound, upper_bound)
    return array

def _calculate_dice(gt, pred, cls=1):
    gt_bin = (gt == cls)
    pred_bin = (pred == cls)
    intersection = np.sum(gt_bin & pred_bin)
    return (2 * intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-8)

def _calculate_iou(gt, pred, cls=1):
    gt_bin = (gt == cls)
    pred_bin = (pred == cls)
    intersection = np.sum(gt_bin & pred_bin)
    union = np.sum(gt_bin | pred_bin)
    return intersection / (union + 1e-8)

def compute_metrics(gt_np, score_np):
    scaled = min_max_scale(replace_outliers_with_bounds(score_np.copy()))
    scaled = (scaled * 255).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask.squeeze() == 255).astype(int)

    metrics = {}
    for cls in [0, 1]:
        metrics[f"dice_class_{cls}"] = _calculate_dice(gt_np, mask, cls)
        metrics[f"iou_class_{cls}"] = _calculate_iou(gt_np, mask, cls)

    TP = np.logical_and(gt_np == 1, mask == 1).sum()
    TN = np.logical_and(gt_np == 0, mask == 0).sum()
    FP = np.logical_and(gt_np == 0, mask == 1).sum()
    FN = np.logical_and(gt_np == 1, mask == 0).sum()

    metrics.update({
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity": TP / (TP + FN + 1e-8),
        "specificity": TN / (TN + FP + 1e-8),
        "precision": TP / (TP + FP + 1e-8),
        "accuracy": (TP + TN) / (TP + TN + FP + FN + 1e-8),
        "f1": 2 * TP / (2 * TP + FP + FN + 1e-8),
        "correct_pixels": np.sum(gt_np == mask),
        "total_pixels": gt_np.size
    })

    return metrics

def main(args):
    config = load_config(args.config)
    folds = config.get("folds", [1, 2, 3, 4, 5])
    methods = config.get("methods", [
        "integrated_gradient",
        "expected_gradient",
        "integrated_decision_gradient",
        "contrastive_gradient",
        "vanilla_gradient",
        "square_integrated_gradient"
    ])

    scores_root = config["attribution_scores_folder"]
    gt_root = config["gt_path"]
    output_csv = config.get("save_csv", "results.csv")
    df = pd.read_csv(config["split_csv"])
    basenames = df["image"].str.replace(".tif", "").values

    all_results = []
    for fold in folds:
        for method in methods:
            for basename in basenames:
                score_file = os.path.join(scores_root, method, f"fold_{fold}", f"{basename}.npy")
                gt_file = os.path.join(gt_root, f"{basename}.npy")

                if not os.path.exists(score_file):
                    print("Missing score:", score_file)
                    continue

                if os.path.exists(gt_file):
                    gt_np = np.load(gt_file)
                else:
                    _type = df[df["image"] == f"{basename}.tif"]["type"].values[0]
                    gt_np = np.zeros((512, 512)) if _type == "normal" else None

                if gt_np is None:
                    print("Missing GT and no default for:", basename)
                    continue

                score_np = np.load(score_file)
                metrics = compute_metrics(gt_np, score_np)

                row = {
                    "fold": fold,
                    "image": basename,
                    "method": method,
                    "type": df[df["image"] == f"{basename}.tif"]["type"].values[0],
                    "class": df[df["image"] == f"{basename}.tif"]["class"].values[0],
                }
                row.update(metrics)
                all_results.append(row)

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    summary = df_out.groupby(["fold", "method", "class"])[
        ["accuracy", "f1", "sensitivity", "specificity", "dice_class_0", "dice_class_1", "iou_class_0", "iou_class_1"]
    ].mean().reset_index()
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved per-class summary to {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    parser.add_argument('--start_fold', type=int, required=True, help="Start fold number")
    parser.add_argument('--end_fold', type=int, required=True, help="End fold number")
    args = parser.parse_args()

    config = load_config(args.config)
    config["folds"] = list(range(args.start_fold, args.end_fold + 1))

    paths = config["paths"]
    config["scores_path"] = paths["attribution_scores_folder"]
    config["gt_path"] = os.path.join(paths["h5_files"], "gt_np")
    config["split_csv"] = os.path.join(paths["split_folder"], f"fold_{args.start_fold}", "test.csv")
    config["save_csv"] = os.path.join(paths["result_dir"], f"results_fold_{args.start_fold}.csv")

    args = argparse.Namespace(config=args.config)
    main(args)