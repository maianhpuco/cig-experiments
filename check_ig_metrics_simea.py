# This script loads a CLAM model, loops through all IG variants, computes attribution,
# and evaluates it using PIC metrics (AIC & SIC) for a single .pt feature example.

import os
import sys
import argparse
import torch
import yaml
import numpy as np

sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("attr_method"))

from clam import load_clam_model


def load_ig_module(args):
    from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

    def get_module_by_name(name):
        if name == 'ig':
            from attr_method.ig import IG as AttrMethod
            print("Using Integrated Gradients (IG) method")
        elif name == 'cig':
            from attr_method.cig import CIG as AttrMethod
            print("Using Cumulative Integrated Gradients (CIG) method")
        elif name == 'idg':
            from attr_method.idg_w_batch import IDG as AttrMethod
            print("Using Integrated Decision Gradients (IDG) method with batch support")
        elif name == 'eg':
            from attr_method.eg import EG as AttrMethod
            print("Using Efficient Expected Gradients (EG) method")
        else:
            raise ValueError(f"Unsupported IG method: {name}")
        return AttrMethod()

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

    return get_module_by_name(args.ig_name), call_model_function


def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def main(args, config):
    import importlib
    sys.path.append(os.path.join("src/evaluation"))
    from PICTestFunctions import compute_pic_metric, generate_random_mask, ComputePicMetricError

    basename = "tumor_028"
    fold_idx = 1
    feature_path = os.path.join(config['paths']['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(config['paths'][f'for_ig_checkpoint_path_fold_{fold_idx}'])
    memmap_path = os.path.join(config['paths']['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)

    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)
    if features.dim() == 3:
        features = features.squeeze(0)
    print(f"> Feature shape: {features.shape}")

    with torch.no_grad():
        output = model(features, [features.shape[0]])
        logits, probs, predicted_class, *_ = output
    pred_class = predicted_class.item()

    features = features.unsqueeze(0)
    mean_vector = features.mean(dim=1, keepdim=True)
    baseline = mean_vector.expand_as(features)

    ig_methods = ['ig', 'cig', 'idg', 'eg']
    # ig_methods = ['eg'] 
    num_patches = features.shape[1]
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]
    random_mask = generate_random_mask(num_patches, fraction=0.01)
    print(f"\n> Number of patches: {num_patches}")
    print("Number of masked patches:", random_mask.sum())
    results_all = {}
    
    saliency_maps = {}
    for ig_name in ig_methods:
        print(f"\n>> Running IG method: {ig_name}")
        args.ig_name = ig_name
        ig_module, call_model_function = load_ig_module(args)

        kwargs = {
            "x_value": features,
            "call_model_function": call_model_function,
            "model": model,
            "baseline_features": baseline,
            "memmap_path": memmap_path,
            "x_steps": 10,
            "device": args.device,
            "call_model_args": {"target_class_idx": pred_class},
            "batch_size": 500
        }
        attribution_values = ig_module.GetMask(**kwargs)
        saliency_map = np.abs(attribution_values).sum(axis=-1).squeeze()

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
            results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
            print(f"  - SIC AUC: {sic_score.auc:.3f}")
            print(f"  - AIC AUC: {aic_score.auc:.3f}")
        except ComputePicMetricError as e:
            print(f"  > Failed for {ig_name}: {e}")
            results_all[ig_name] = None

    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        if v:
            print(f"{k.upper():<5} : SIC = {v['SIC']:.3f} | AIC = {v['AIC']:.3f}")
        else:
            print(f"{k.upper():<5} : FAILED")

# Print correlations between saliency maps
    from scipy.stats import pearsonr

    print("\n=== Saliency Map Correlations ===")
    for m1 in ig_methods:
        for m2 in ig_methods:
            if m1 < m2 and m1 in saliency_maps and m2 in saliency_maps:
                corr, _ = pearsonr(saliency_maps[m1], saliency_maps[m2])
                print(f"{m1.upper()} vs {m2.upper()}: Pearson correlation = {corr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Configuration Loaded ===")
    print(f"> Device       : {args.device}")
    print(f"> Dropout      : {args.drop_out}")
    print(f"> Embed dim    : {args.embed_dim}")
    print(f"> Model type   : {args.model_type}")

    main(args, config)
