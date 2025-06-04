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


def compute_one_slide(args, basename, model):
    fold_id = args.fold

    if args.dataset_name == 'camelyon16':
        feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
    elif args.dataset_name == 'tcga_renal':
        slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
        true_label = int(slide_row['true_label'].iloc[0]) if not slide_row.empty else -1
        reverse_label_dict = {v: k for k, v in args.label_dict.items()}
 
        print("label dict", reverse_label_dict) 
        if true_label == -1:
            raise ValueError(f"True label not found for slide {basename}")
        subtype = reverse_label_dict[true_label].lower()
        feature_path = os.path.join(args.paths['data_dir'][subtype], f"pt_files/{basename}.pt")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

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

    if baseline.shape[0] != features.shape[0]:
        print(f"[WARN] Baseline patch count ({baseline.shape[0]}) doesn't match features ({features.shape[0]}). Resampling baseline.")
        indices = torch.randint(0, baseline.shape[0], (features.shape[0],), device=baseline.device)
        baseline = baseline[indices]

    baseline_pred = model(baseline)
    _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1)
    print(f"> Baseline predicted class: {baseline_predicted_class.item()}")
    print(f"> Baseline logits: {baseline_pred[0].detach().cpu().numpy()}")

    print("========== LOAD PRECOMPUTED ATTRIBUTION ==========")
    ig_name = args.ig_name
    attribution_path = os.path.join(
        args.paths['attribution_scores_folder'], f"fold_{fold_id}", ig_name, f"{basename}.npy"
    )
    if not os.path.isfile(attribution_path):
        raise FileNotFoundError(f"Attribution map not found: {attribution_path}")
    attribution_values = np.load(attribution_path)

    saliency_map = np.mean(np.abs(attribution_values), axis=-1).squeeze()
    saliency_map = saliency_map / (saliency_map.max() + 1e-8)

    print(f"  - Saliency map shape: {saliency_map.shape} Stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")

    top_k = np.array(
        list(range(1, 11)) +
        list(range(15, 55, 5)) +
        list(range(60, 110, 10)) +
        [150, 200, 300, 400, 500]
    )
    random_mask = generate_random_mask(features.shape[0], fraction=0.0)

    slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
    pred_label = slide_row['pred_label'].iloc[0] if not slide_row.empty else pred_class
    true_label = slide_row['true_label'].iloc[0] if 'true_label' in slide_row.columns else -1

    sic_score = compute_pic_metric(top_k, features.cpu().numpy(), saliency_map, random_mask,
                                None, 0, model, args.device,
                                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                                keep_monotonous=False)
    aic_score = compute_pic_metric(top_k, features.cpu().numpy(), saliency_map, random_mask,
                                None, 1, model, args.device,
                                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                                keep_monotonous=False)

    result = {
        "slide_id": basename,
        "pred_label": pred_label,
        "true_label": true_label,
        "baseline_pred_label": baseline_predicted_class.item(),
        "saliency_map_mean": saliency_map.mean(),
        "saliency_map_std": saliency_map.std(),
        "IG": ig_name,
        "AIC": aic_score.auc,
        "SIC": sic_score.auc
    }

    print(f"\n> Result: {result}")
    return result


def main(args):
    fold_id = args.fold = 1
    # checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'])
    checkpoint_path = args.ckpt_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    
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

    print(f"[INFO] Loaded {len(tumor_df)} tumor slides from predictions")

    all_results = []
    start = time.time()
    for idx, basename in enumerate(basenames):
        print(f"\n=== Processing slide: {basename} ({idx + 1}/{len(basenames)}) ===")
        all_results.append(compute_one_slide(args, basename, model))

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
