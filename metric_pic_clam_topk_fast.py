import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.extend([
    os.path.join("src/models"),
    os.path.join("src/models/classifiers"),
    os.path.join("src/attr_method"),
    os.path.join("src/evaluation")
])

from clam import load_clam_model
from PICTestFunctions import compute_pic_metric, generate_random_mask, ModelWrapper

def parse_args_from_config(config):
    class ConfigArgs:
        pass

    args = ConfigArgs()
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)
    args.device = getattr(args, 'device', "cuda" if torch.cuda.is_available() else "cpu")
    return args

def compute_one_slide(args, basename, model, baseline_dict, attribution_dict):
    fold_id = args.fold

    if args.dataset_name == 'camelyon16':
        feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
        true_label = int(slide_row['true_label'].iloc[0]) if not slide_row.empty else -1
        reverse_label_dict = {v: k for k, v in args.label_dict.items()}
        subtype = reverse_label_dict[true_label].lower()
        feature_path = os.path.join(args.paths['data_dir'][subtype], f"pt_files/{basename}.pt")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32).squeeze(0)
    features_data = features.unsqueeze(0) if features.dim() == 2 else features

    with torch.no_grad():
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(features.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, dim=1)
    pred_class = predicted_class.item()

    baseline = baseline_dict[basename].to(args.device, dtype=torch.float32).squeeze(0)
    if baseline.shape[0] != features.shape[0]:
        indices = torch.randint(0, baseline.shape[0], (features.shape[0],), device=baseline.device)
        baseline = baseline[indices]

    with torch.no_grad():
        baseline_pred = model(baseline)
        _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1)

    attribution_values = attribution_dict[basename]
    saliency_map = np.mean(np.abs(attribution_values), axis=-1).squeeze()
    saliency_map = saliency_map / (saliency_map.max() + 1e-8)

    top_k = np.array(list(range(1, 11)) + list(range(15, 55, 5)) + list(range(60, 110, 10)) + [150, 200, 300, 400, 500])
    random_mask = generate_random_mask(features.shape[0], fraction=0.0)

    slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
    pred_label = slide_row['pred_label'].iloc[0] if not slide_row.empty else pred_class
    true_label = slide_row['true_label'].iloc[0] if 'true_label' in slide_row.columns else -1

    features_np = features.cpu().numpy()
    baseline_np = baseline.cpu().numpy()

    sic_score = compute_pic_metric(top_k, features_np, saliency_map, random_mask,
                                   None, 0, model, args.device,
                                   baseline=baseline_np, min_pred_value=0.3,
                                   keep_monotonous=False)

    aic_score = compute_pic_metric(top_k, features_np, saliency_map, random_mask,
                                   None, 1, model, args.device,
                                   baseline=baseline_np, min_pred_value=0.3,
                                   keep_monotonous=False)

    return {
        "slide_id": basename,
        "pred_label": pred_label,
        "true_label": true_label,
        "baseline_pred_label": baseline_predicted_class.item(),
        "saliency_map_mean": saliency_map.mean(),
        "saliency_map_std": saliency_map.std(),
        "IG": args.ig_name,
        "AIC": aic_score.auc,
        "SIC": sic_score.auc
    }

def main(args):
    fold_id = args.fold = 1
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}")

    args_clam = argparse.Namespace(
        drop_out=args.drop_out,
        n_classes=args.n_classes,
        embed_dim=args.embed_dim,
        model_type=args.model_type,
        model_size=args.model_size
    )

    model = load_clam_model(args_clam, args.ckpt_path, device=args.device)
    model.eval()

    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    args.pred_df = pred_df
    basenames = pred_df['slide_id'].unique().tolist()
    print(f"[INFO] Loaded {len(pred_df)} slides")

    # Preload baselines and attributions
    baseline_dir = args.paths[f'baseline_dir_fold_{fold_id}']
    attribution_dir = os.path.join(args.paths['attribution_scores_folder'], args.ig_name, f"fold_{fold_id}")

    baseline_dict = {bn: torch.load(os.path.join(baseline_dir, f"{bn}.pt")) for bn in basenames}
    attribution_dict = {bn: np.load(os.path.join(attribution_dir, f"{bn}.npy")) for bn in basenames}

    all_results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(compute_one_slide, args, bn, model, baseline_dict, attribution_dict): bn for bn in basenames}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed on slide {futures[future]}: {e}")

    results_df = pd.DataFrame(all_results)
    output_dir = os.path.join(args.paths['metrics_dir'], f"{args.ig_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"topk_pic_results_fold_{fold_id}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n> Results saved to: {output_path}")
    avg_results = results_df.groupby("IG")[["AIC", "SIC"]].mean().reset_index()
    print("\n=== Average AIC and SIC per IG Method ===")
    print(avg_results.to_string(index=False))
    print(f"\n> Total time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ig_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args_from_config(config)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.ig_name = args_cmd.ig_name
    args.ckpt_path = args_cmd.ckpt_path

    main(args)
