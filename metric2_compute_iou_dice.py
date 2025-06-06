import os
import argparse
import numpy as np
import pandas as pd
import yaml
import time
from tqdm import tqdm
from skimage.filters import threshold_otsu


def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)


def iou_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)


def compute_one_ig(args):
    fold_id = args.fold if hasattr(args, 'fold') else 1
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    args.pred_df = pred_df
    basenames = pred_df['slide_id'].unique().tolist()

    print(f"[INFO] Loaded predictions for {len(basenames)} slides")

    results = []
    start = time.time()

    for idx, basename in enumerate(tqdm(basenames, desc="Processing slides", ncols=100)):
        row = pred_df[pred_df['slide_id'] == basename].iloc[0]
        if row['true_label'] != 1 or row['pred_label'] != 1:
            continue  # only include correctly predicted tumor

        gt_mask_path = os.path.join(args.paths['ground_truth_numpy_dir'], f"{basename}.npy")
        score_path = os.path.join(args.paths['attribution_scores_folder'], args.ig_name, f"fold_{fold_id}", f"{basename}.npy")

        if not os.path.exists(gt_mask_path) or not os.path.exists(score_path):
            continue

        gt_mask = np.load(gt_mask_path)
        scores = np.load(score_path).squeeze()

        if scores.ndim == 2:
            # scores = np.mean(scores, axis=-1)
            scores = np.mean(np.abs(scores), axis=-1).squeeze()
 
        print(">>>> score shape", scores.shape)
        clipped_scores = np.clip(scores, np.percentile(scores, 1), np.percentile(scores, 99))
        scaled_scores = (clipped_scores - clipped_scores.min()) / (clipped_scores.max() - clipped_scores.min() + 1e-8)

        try:
            threshold = threshold_otsu(scaled_scores)
        except:
            continue

        pred_mask = (scaled_scores > threshold).astype(np.uint8)
        gt_tumor = (gt_mask == 1).astype(np.uint8)
        gt_normal = (gt_mask == 0).astype(np.uint8)
        pred_normal = 1 - pred_mask

        print("> Sum gt_tumor", gt_tumor.sum())
        print("> Sum gt_normal", gt_normal.sum())
        
        print("> Sum gt_tumor", gt_tumor.sum())
        print("> Sum gt_normal", gt_normal.sum())  
        
        dice_tumor = dice_score(gt_tumor, pred_mask)
        iou_tumor = iou_score(gt_tumor, pred_mask)
        dice_normal = dice_score(gt_normal, pred_normal)
        iou_normal = iou_score(gt_normal, pred_normal)
        result = {
            "slide_id": basename,
            "dice_tumor": dice_tumor,
            "iou_tumor": iou_tumor,
            "dice_normal": dice_normal,
            "iou_normal": iou_normal,
            "threshold": threshold,
            "gt_sum": int(gt_tumor.sum()),
            "pred_sum": int(pred_mask.sum())
        }
        results.append(result)
        print(result)
        # return 

    df_result = pd.DataFrame(results)
    output_dir = os.path.join(args.paths['dice_iou_dir'], f"{args.ig_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dice_iou_fold_{fold_id}.csv")
    df_result.to_csv(output_path, index=False)

    print(f"\n✅ Saved results to: {output_path}")

    if not df_result.empty:
        print("\n📊 AVERAGE METRICS:")
        for metric in ["dice_tumor", "iou_tumor", "dice_normal", "iou_normal"]:
            avg = df_result[metric].mean()
            std = df_result[metric].std()
            print(f"{metric}: {avg:.4f} ± {std:.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️ Total computation time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ig_name', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    compute_one_ig(args)
