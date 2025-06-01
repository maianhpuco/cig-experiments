# This script loads a CLAM model, computes Integrated Gradients-based attribution
# for a single example using a .pt feature file, and evaluates it using PIC metrics (AIC & SIC)

import os
import sys
import argparse
import torch
import yaml
import numpy as np

# Add paths for model and attribution code
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("attr_method"))

from clam import load_clam_model

# Load Integrated Gradients module dynamically based on user argument
def load_ig_module(args):
    from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
    if args.ig_name == 'ig':
        from attr_method.ig import IG as AttrMethod
    elif args.ig_name == 'cig':
        from attr_method.cig import CIG as AttrMethod
    elif args.ig_name == 'idg':
        from attr_method.idg_w_batch import IDG as AttrMethod
    elif args.ig_name == 'eg':
        from attr_method.eg import EG as AttrMethod
    else:
        print("> Error: Unsupported attribution method name.")

    # Custom wrapper function for CLAM forward pass and gradient capture
    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3

        inputs = inputs.to(device).clone().detach().requires_grad_(True)
        if was_batched:
            inputs = inputs.squeeze(0)

        model.eval()
        outputs = model(inputs, [inputs.shape[0]])
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
            class_idx = call_model_args.get("target_class_idx", 0)
            target = logits[:, class_idx]
            grads = torch.autograd.grad(
                outputs=target,
                inputs=inputs,
                grad_outputs=torch.ones_like(target),
                retain_graph=False,
                create_graph=False
            )[0]
            grads_np = grads.detach().cpu().numpy()
            if was_batched or grads_np.ndim == 2:
                grads_np = np.expand_dims(grads_np, axis=0)
            return {INPUT_OUTPUT_GRADIENTS: grads_np}

        return logits

    ig = AttrMethod()
    return ig, call_model_function

# Setup default CLAM model hyperparameters

def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def main(args, config):
    # Setup feature/checkpoint/memmap paths from YAML
    basename = "tumor_028"
    fold_idx = 1
    feature_path = os.path.join(config['paths']['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(config['paths'][f'for_ig_checkpoint_path_fold_{fold_idx}'])
    memmap_path = os.path.join(config['paths']['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    # Load CLAM model
    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)

    # Load attribution method
    ig_module, call_model_function = load_ig_module(args)

    # Load feature file (patch-level feature vectors)
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)
    if features.dim() == 3:
        features = features.squeeze(0)
    print(f"> Feature shape: {features.shape}")

    # Predict class
    with torch.no_grad():
        output = model(features, [features.shape[0]])
        logits, probs, predicted_class, *_ = output
    pred_class = predicted_class.item()
    print(f"\n> Prediction Complete\n  - Logits: {logits}\n  - Probabilities: {probs}\n  - Predicted class: {pred_class}")

    # Create baseline (mean vector repeated for all patches)
    features = features.unsqueeze(0)  # [1, N, D]
    mean_vector = features.mean(dim=1, keepdim=True)  # [1, 1, D]
    baseline = mean_vector.expand_as(features)        # [1, N, D]

    print(f"> Feature shape  : {features.shape}")
    print(f"> Baseline shape : {baseline.shape}")

    # Compute saliency using IG variant
    kwargs = {
        "x_value": features,
        "call_model_function": call_model_function,
        "model": model,
        "baseline_features": baseline,
        "memmap_path": memmap_path,
        "x_steps": 5,
        "device": args.device,
        "call_model_args": {"target_class_idx": pred_class},
        "batch_size": 500
    }
    attribution_values = ig_module.GetMask(**kwargs)
    saliency_map = attribution_values.mean(1)
    print(f"  - Attribution shape: {attribution_values.shape}\n  - Mean score shape : {saliency_map.shape}")

    # Compute AIC and SIC using PIC metrics
    sys.path.append(os.path.join("src/evaluation"))
    from PICTestFunctions import compute_pic_metric, generate_random_mask, ComputePicMetricError

    num_patches = features.shape[1]
    random_mask = generate_random_mask(num_patches, fraction=0.01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    try:
        sic_score = compute_pic_metric(
            features=features.squeeze().cpu().numpy(),
            saliency_map=saliency_map,
            random_mask=random_mask,
            saliency_thresholds=saliency_thresholds,
            method=0,  # SIC
            model=model,
            device=args.device,
            min_pred_value=0.8
        )
        aic_score = compute_pic_metric(
            features=features.squeeze().cpu().numpy(),
            saliency_map=saliency_map,
            random_mask=random_mask,
            saliency_thresholds=saliency_thresholds,
            method=1,  # AIC
            model=model,
            device=args.device,
            min_pred_value=0.8
        )
        print(f"\n> PIC Metrics Computed\n  - SIC AUC: {sic_score.auc:.3f}\n  - AIC AUC: {aic_score.auc:.3f}")

        # Save results to YAML
        results = {
            "wsi_file": os.path.basename(feature_path),
            "predicted_class": pred_class,
            "sic_auc": sic_score.auc,
            "aic_auc": aic_score.auc
        }
        output_file = os.path.join(memmap_path, "pic_results.yaml")
        with open(output_file, 'w') as f:
            yaml.safe_dump(results, f)
        print(f"> Results saved to: {output_file}")

    except ComputePicMetricError as e:
        print(f"> Failed to compute PIC metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Load YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update args from YAML
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.ig_name = 'ig'  # Select IG variant to run (e.g., ig, cig, idg, eg)

    print("=== Configuration Loaded ===")
    print(f"> Device       : {args.device}")
    print(f"> Dropout      : {args.drop_out}")
    print(f"> Embed dim    : {args.embed_dim}")
    print(f"> Model type   : {args.model_type}")

    main(args, config)
