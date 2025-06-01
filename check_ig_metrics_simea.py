import os
import sys
import argparse
import torch
import yaml
import numpy as np
from scipy.stats import pearsonr

# Add paths for model, attribution, and evaluation code
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("src/attr_method"))
sys.path.append(os.path.join("src/evaluation"))

from clam import load_clam_model
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from PICTestFunctions import compute_pic_metric, generate_random_mask, ComputePicMetricError, ModelWrapper

# Inline IDGWrapper for IDG instantiation
class IDGWrapper(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")
        model = kwargs.get("model")
        baseline_features = kwargs.get("baseline_features")
        x_steps = kwargs.get("x_steps", 10)
        device = kwargs.get("device", "cpu")
        call_model_args = kwargs.get("call_model_args", {})
        batch_size = kwargs.get("batch_size", 500)
        target_class = call_model_args.get("target_class_idx", 0)

        if x_value.dim() == 3:
            x_value = x_value.squeeze(0)

        from attr_method.idg_w_batch import IDG
        attribution = IDG(
            input=x_value,
            model=model,
            steps=x_steps,
            batch_size=batch_size,
            baseline=baseline_features,
            device=device,
            target_class=target_class
        )

        if attribution.dim() == 2:
            attribution = attribution.unsqueeze(0)

        return attribution.cpu().numpy()

def load_ig_module(args):
    def get_module_by_name(name):
        if name == 'ig':
            from attr_method.ig import IG as AttrMethod
            print("Using Integrated Gradients (IG) method")
            return AttrMethod()
        elif name == 'cig':
            from attr_method.cig import CIG as AttrMethod
            print("Using Cumulative Integrated Gradients (CIG) method")
            return AttrMethod()
        elif name == 'idg':
            print("Using Integrated Decision Gradients (IDG) method with batch support")
            return IDGWrapper()
        elif name == 'eg':
            from attr_method.eg import EG as AttrMethod
            print("Using Expected Gradients (EG) method")
            return AttrMethod()
        else:
            raise ValueError(f"Unsupported IG method: {name}")

    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3
        inputs = inputs.to(device).clone().detach().requires_grad_(True)
        if was_batched:
            inputs = inputs.squeeze(0)
        model.eval()
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(inputs)

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
    basename = "tumor_029"
    fold_idx = 1
    feature_path = os.path.join(config['paths']['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(config['paths'][f'for_ig_checkpoint_path_fold_{fold_idx}'])
    memmap_path = os.path.join(config['paths']['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    # Load dataset mean and variance
    mean_std_path = os.path.join("/home/mvu9/processing_datasets/processing_camelyon16/dataset_mean_variance", f"fold_{fold_idx}_mean_std.pt")
    print(f"\n> Loading dataset mean and variance from: {mean_std_path}")
    try:
        mean_std_data = torch.load(mean_std_path)
        dataset_mean = mean_std_data['mean'].cpu().numpy()  # Shape: [D,]
        dataset_std = mean_std_data['std'].cpu().numpy()  # Shape: [D,]
        print(f"> Dataset mean shape: {dataset_mean.shape}, std shape: {dataset_std.shape}")
        print(f"> Dataset std range: min={dataset_std.min():.6f}, max={dataset_std.max():.6f}")
    except Exception as e:
        print(f"> Failed to load mean and variance: {e}")
        return

    # Load CLAM model
    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)
    model.eval()

    # Load feature file
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)
    if features.dim() == 3:
        features = features.squeeze(0)
    print(f"> Feature shape: {features.shape}")

    # Predict class
    with torch.no_grad():
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(features.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, dim=1)
    pred_class = predicted_class.item()
    print(f"\n> Prediction Complete\n  - Logits: {logits}\n  - Probabilities: {probs}\n  - Predicted class: {pred_class}")

    # Create baseline by sampling from normal distribution
    features = features.unsqueeze(0)  # [1, N, D]
    num_patches = features.shape[1]
    embed_dim = features.shape[2]
    mean_tensor = torch.from_numpy(dataset_mean).to(args.device, dtype=torch.float32)  # [D,]
    std_tensor = torch.from_numpy(dataset_std).to(args.device, dtype=torch.float32)  # [D,]
    dist = torch.distributions.Normal(mean_tensor, std_tensor * 2 + 1e-6)  # Scale std and add epsilon
    baseline = dist.sample((1, num_patches)).to(args.device, dtype=torch.float32)  # [1, N, D]
    print(f"> Feature shape  : {features.shape}")
    print(f"> Baseline shape : {baseline.shape}")
    print(f"> Baseline stats : mean={baseline.mean().item():.6f}, std={baseline.std().item():.6f}")

    # IG methods to evaluate
    ig_methods = ['ig', 'cig', 'idg', 'eg']
    saliency_thresholds = np.linspace(0.005, 0.75, 20)  # Finer thresholds
    random_mask = generate_random_mask(num_patches, fraction=0.01)
    print(f"\n> Number of patches: {num_patches}")
    print(f"> Number of masked patches: {random_mask.sum()}")

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
        try:
            attribution_values = ig_module.GetMask(**kwargs)
            saliency_map = np.abs(np.mean(np.abs(attribution_values), axis=-1)).squeeze()  # [N,]
            # Normalize saliency map
            saliency_map = saliency_map / (saliency_map.max() + 1e-8)  # Normalize to [0, 1]
            saliency_maps[ig_name] = saliency_map
            print(f"  - Attribution shape: {attribution_values.shape}")
            print(f"  - Saliency map shape: {saliency_map.shape}")
            print(f"  - Saliency map stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}, min={saliency_map.min():.6f}, max={saliency_map.max():.6f}")
            print(f"  - Saliency map unique values: {np.unique(saliency_map).size}")

            sic_score = compute_pic_metric(
                features=features.squeeze().cpu().numpy(),
                saliency_map=saliency_map,
                random_mask=random_mask,
                saliency_thresholds=saliency_thresholds,
                method=0,  # SIC
                model=model,
                device=args.device,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                min_pred_value=0.3,  # Lowered further
                keep_monotonous=False
            )
            aic_score = compute_pic_metric(
                features=features.squeeze().cpu().numpy(),
                saliency_map=saliency_map,
                random_mask=random_mask,
                saliency_thresholds=saliency_thresholds,
                method=1,  # AIC
                model=model,
                device=args.device,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                min_pred_value=0.3,
                keep_monotonous=False
            )
            results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
            print(f"  - SIC AUC: {sic_score.auc:.3f}")
            print(f"  - AIC AUC: {aic_score.auc:.3f}")
        except Exception as e:
            print(f"  > Failed for {ig_name}: {e}")
            results_all[ig_name] = None

    # Print correlations between saliency maps
    print("\n=== Saliency Map Correlations ===")
    for m1 in ig_methods:
        for m2 in ig_methods:
            if m1 < m2 and m1 in saliency_maps and m2 in saliency_maps:
                corr, _ = pearsonr(saliency_maps[m1], saliency_maps[m2])
                print(f"{m1.upper()} vs {m2.upper()}: Pearson correlation = {corr:.3f}")

    # Print summary of results
    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        if v:
            print(f"{k.upper():<5} : SIC = {v['SIC']:.3f} | AIC = {v['AIC']:.3f}")
        else:
            print(f"{k.upper():<5} : FAILED")

    # Save results
    output_file = os.path.join(memmap_path, "pic_results.yaml")
    with open(output_file, 'w') as f:
        yaml.safe_dump({
            "wsi_file": os.path.basename(feature_path),
            "predicted_class": pred_class,
            "results": results_all
        }, f)
    print(f"> Results saved to: {output_file}")

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