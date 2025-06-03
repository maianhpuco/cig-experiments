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
from PICTestFunctions import compute_pic_metric, generate_random_mask, ModelWrapper
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
            from attr_method.idg_w_batch import IDG as AttrMethod 
            print("Using Integrated Decision Gradients (IDG) method with batch support")
            return AttrMethod()
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

def parse_args_from_config(config):
    class ConfigArgs:
        pass
    args = ConfigArgs()
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)
    args.device = args.device if hasattr(args, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")
    return args

def main(args):
    basename = 'test_001'
    fold_id = 1
    
    feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'])
    memmap_path = os.path.join(args.paths['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    from argparse import Namespace
    args_clam = Namespace(
        drop_out=args.drop_out,
        n_classes=args.n_classes,
        embed_dim=args.embed_dim,
        model_type=args.model_type,
        model_size=args.model_size
    )
    
    
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)
    model.eval()
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
    # Load saved baseline
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline file not found at: {baseline_path}")
    baseline = torch.load(baseline_path).to(args.device, dtype=torch.float32)
      
    print(f"> Loaded baseline from: {baseline_path} | Shape: {baseline.shape}")

    baseline = baseline.squeeze(0) if baseline.dim() == 3 else baseline
    baseline_pred = model(baseline)
    _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1) 
    print(f"> Baseline predicted class: {baseline_predicted_class.item()}")
    print(f"> Baseline logits: {baseline_pred[0].detach().cpu().numpy()}")
    # print(f"> Baseline prediction: {baseline_pred}")
    print("========== SANITY CHECK : Compare Baseline and Features ==========")
    cosine_similarity = torch.nn.functional.cosine_similarity(features, baseline, dim=1)
    print(f"> Cosine similarity between features and baseline: {cosine_similarity.mean().item():.4f}")
    # Replace 5% of baseline with actual features
    num_patches = baseline.shape[0]
    num_replace = max(1, int(num_patches * 0.0005))  # Ensure at least one patch is replaced

    replace_indices = torch.randperm(num_patches)[:num_replace]
    baseline_modified = baseline.clone()
    baseline_modified[replace_indices] = features[replace_indices]

    # Predict using modified baseline
    modified_pred = model(baseline_modified)
    _, modified_predicted_class = torch.max(modified_pred[0], dim=1)

    print("\n========== PREDICTION FOR MODIFIED BASELINE (5% real features) ==========")
    print(f"> Replaced indices: {len(replace_indices.tolist())}")
    print(f"> Modified predicted class: {modified_predicted_class.item()}")
    print(f"> Modified logits: {modified_pred[0].detach().cpu().numpy()}")

    # Cosine similarity to compare modified baseline with original features
    cosine_similarity_modified = torch.nn.functional.cosine_similarity(features, baseline_modified, dim=1)
    print(f"> Cosine similarity (modified vs. features): {cosine_similarity_modified.mean().item():.4f}")

    
    print("==========COMPUTE IG METHODS ==========")
    # IG methods
    # ig_methods = ['ig', 'cig', 'idg', 'eg']
    ig_methods = ['ig', 'random']  # For testing, only use IG and ranran
    # saliency_thresholds = np.linspace(0.01, 0.99, 10)
    # saliency_thresholds = np.logspace(-3, np.log10(0.6), num=12, base=10.0)
    # saliency_thresholds = np.logspace(-4, np.log10(0.5), num=15, base=10.0)
    # saliency_thresholds = np.array([0.00005, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04])
    saliency_thresholds = np.array([
        0.00001, 0.00002, 0.00005, 0.0001, 0.0002,
        0.0005, 0.001, 0.0015, 0.002, 0.003,
        0.004, 0.005, 0.0075, 0.01, 0.015,
        0.02, 0.03, 0.04, 0.05, 0.075
    ])
    saliency_thresholds = np.clip(saliency_thresholds, 0, 1)  # Ensure all values are within [0, 1]

    print("> Saliency thresholds:", saliency_thresholds)
    
    # saliency_thresholds = np.linspace(0.005, 0.95, 10)
    print(f"\n> Saliency thresholds: {saliency_thresholds}")
    random_mask = generate_random_mask(features.shape[0], fraction=0)
    print(f"\n> Number of patches: {features.shape[0]}")
    print(f"> Number of masked patches: {random_mask.sum()}")

    results_all = {}
    saliency_maps = {}
    
    for ig_name in ig_methods:
        print(f"\n>> Running IG method: {ig_name}")
        if ig_name == 'random':
            # Generate random saliency map
            saliency_map = np.random.rand(features.shape[0])
            saliency_map = saliency_map / (saliency_map.max() + 1e-8)
            saliency_maps[ig_name] = saliency_map
            print(f"  - Random saliency stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")

        else:
            args.ig_name = ig_name
            ig_module, call_model_function = load_ig_module(args)
            kwargs = {
                "x_value": features_data,
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
                saliency_map = np.abs(np.mean(np.abs(attribution_values), axis=-1)).squeeze()
                saliency_map = saliency_map / (saliency_map.max() + 1e-8)
                saliency_maps[ig_name] = saliency_map
                print(f"  - Saliency stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")
            except Exception as e:
                print(f"  > Failed for {ig_name}: {e}")
                results_all[ig_name] = None
                continue

        try:
            sic_score = compute_pic_metric(
                features.cpu().numpy(), saliency_map, random_mask,
                saliency_thresholds, 0, model, args.device,
                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                keep_monotonous=False
            )
            aic_score = compute_pic_metric(
                features.cpu().numpy(), saliency_map, random_mask,
                saliency_thresholds, 1, model, args.device,
                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                keep_monotonous=False
            )

            results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
            print(f"  - SIC AUC: {sic_score.auc:.3f}\n  - AIC AUC: {aic_score.auc:.3f}")
        except Exception as e:
            print(f"  > Failed to compute PIC for {ig_name}: {e}")
            results_all[ig_name] = None

    # print("\n=== Saliency Map Correlations ===")
    # for m1 in ig_methods:
    #     for m2 in ig_methods:
    #         if m1 < m2 and m1 in saliency_maps and m2 in saliency_maps:
    #             corr, _ = pearsonr(saliency_maps[m1], saliency_maps[m2])
    #             print(f"{m1.upper()} vs {m2i.upper()}: Pearson correlation = {corr:.3f}")

    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        if v:
            print(f"{k.upper():<5} : SIC = {v['SIC']:.3f} | AIC = {v['AIC']:.3f}")
        else:
            print(f"{k.upper():<5} : FAILED")

    output_file = os.path.join(memmap_path, "pic_results.yaml")
    print(f"> Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args_from_config(config)
    main(args)
