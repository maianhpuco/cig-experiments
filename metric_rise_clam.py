import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Extend sys path to access internal modules
sys.path.extend([
    os.path.join("src/models"),
    os.path.join("src/models/classifiers"),
    os.path.join("src/attr_method"),
    os.path.join("src/evaluation")
])

from clam import load_clam_model
from PICTestFunctions import ModelWrapper
from RISETestFunctions import CausalMetric, auc

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

def compute_one_slide(args, basename, model):
    fold_id = args.fold
    feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
    memmap_path = os.path.join(args.paths['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    print("========== PREDICTION FOR FEATURES ==========")
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)
    features = features.squeeze(0) if features.dim() == 3 else features
    features_data = features.unsqueeze(0) if features.dim() == 2 else features

    print(f"> Slide name: {basename}\n> Fold ID: {fold_id}")
    print(f"> Feature shape: {features.shape}")

    with torch.no_grad():
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(features.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, dim=1)
    pred_class = predicted_class.item()

    print(f"\n> Prediction Complete\n  - Logits: {logits}\n  - Probabilities: {probs}\n  - Predicted class: {pred_class}")

    print("========== PREDICTION FOR BASELINE ==========")
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline file not found at: {baseline_path}")
    baseline = torch.load(baseline_path).to(args.device, dtype=torch.float32)
    baseline = baseline.squeeze(0) if baseline.dim() == 3 else baseline

    # Resample baseline to match number of patches
    if baseline.shape[0] != features.shape[0]:
        print(f"[WARN] Baseline patch count ({baseline.shape[0]}) doesn't match features ({features.shape[0]}). Resampling baseline.")
        indices = torch.randint(0, baseline.shape[0], (features.shape[0],), device=baseline.device)
        baseline = baseline[indices]

    baseline_pred = model(baseline)
    _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1)
    print(f"> Baseline predicted class: {baseline_predicted_class.item()}")
    print(f"> Baseline logits: {baseline_pred[0].detach().cpu().numpy()}")

    print("========== LOAD PRECOMPUTED ATTRIBUTION ==========")
    ig_name = "ig"
    attribution_path = os.path.join(
        args.paths['attribution_scores_folder'], f"fold_{fold_id}", ig_name, f"{basename}.npy"
    )
    if not os.path.isfile(attribution_path):
        raise FileNotFoundError(f"Attribution map not found: {attribution_path}")
    attribution_values = np.load(attribution_path)

    saliency_map = np.mean(np.abs(attribution_values), axis=-1).squeeze()
    saliency_map = saliency_map / (saliency_map.max() + 1e-8)

    print(f"  - Saliency map shape: {saliency_map.shape} Stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")

    # Define substrate functions for RISE
    def mean_substrate_fn(feature_tensor):
        mean_features = feature_tensor.mean(dim=1, keepdim=True).expand_as(feature_tensor)
        return mean_features

    def zero_substrate_fn(feature_tensor):
        return torch.zeros_like(feature_tensor)

    # Compute RISE metrics
    print("========== COMPUTE RISE METRICS ==========")
    num_patches = features.shape[0]
    step = max(1, num_patches // 100)  # Adjust step size based on number of patches

    deletion_metric = CausalMetric(model, num_patches, mode='del', step=step, substrate_fn=zero_substrate_fn)
    insertion_metric = CausalMetric(model, num_patches, mode='ins', step=step, substrate_fn=zero_substrate_fn)

    n_steps_del, scores_del = deletion_metric.single_run(features_data, saliency_map, args.device)
    n_steps_ins, scores_ins = insertion_metric.single_run(features_data, saliency_map, args.device)

    sic_score = auc(scores_del) if n_steps_del > 0 else 0.0
    aic_score = auc(scores_ins) if n_steps_ins > 0 else 0.0

    slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
    pred_label = slide_row['pred_label'].iloc[0] if not slide_row.empty else pred_class
    true_label = slide_row['true_label'].iloc[0] if 'true_label' in slide_row.columns else -1

    result = {
        "slide_id": basename,
        "pred_label": pred_label,
        "true_label": true_label,
        "baseline_pred_label": baseline_predicted_class.item(),
        "saliency_map_mean": saliency_map.mean(),
        "saliency_map_std": saliency_map.std(),
        "IG": ig_name,
        "AIC": aic_score,
        "SIC": sic_score
    }

    print(f"\n> Result: {result}")
    return result

def main(args):
    fold_id = args.fold = 1
    checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'])

    args_clam = argparse.Namespace(
        drop_out=args.drop_out,
        n_classes=args.n_classes,
        embed_dim=args.embed_dim,
        model_type=args.model_type,
        model_size=args.model_size
    )

    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)
    model.eval()

    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    tumor_df = pred_df[pred_df['pred_label'] == 1]

    basenames = tumor_df['slide_id'].unique().tolist()
    args.pred_df = tumor_df
    basenames= ['test_001']
    
    print(f"[INFO] Loaded {len(tumor_df)} tumor slides from predictions")

    all_results = []
    start = time.time()
    for idx, basename in enumerate(basenames):
        print(f"\n=== Processing slide: {basename} ({idx + 1}/{len(basenames)}) ===")
        all_results.append(compute_one_slide(args, basename, model))

    results_df = pd.DataFrame(all_results)
    output_dir = os.path.join(args.paths['metrics_dir'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rise_results_fold{fold_id}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n> Results saved to: {output_path}")
    avg_results = results_df.groupby("IG")[["AIC", "SIC"]].mean().reset_index()
    print("\n=== Average AIC and SIC per IG Method ===")
    print(avg_results.to_string(index=False))
    print(f"\n> Total time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args_from_config(config)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    main(args)