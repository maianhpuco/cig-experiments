import os
import sys
import argparse
import torch
import yaml
import numpy as np
from scipy.stats import pearsonr

sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("src/attr_method"))
sys.path.append(os.path.join("src/evaluation"))

from clam import load_clam_model
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from PICTestFunctions2 import compute_pic_metric, generate_random_mask, ModelWrapper
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
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args

def compute_one_slide(args, basename):
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

    print("========== SANITY CHECK : Compare Baseline and Features ==========")
    cosine_similarity = torch.nn.functional.cosine_similarity(features, baseline, dim=1)
    print(f"> Cosine similarity between features and baseline: {cosine_similarity.mean().item():.4f}")
    
    num_patches = baseline.shape[0]
    num_replace = max(1, int(num_patches * 0.0005))
    replace_indices = torch.randperm(num_patches)[:num_replace]
    baseline_modified = baseline.clone()
    baseline_modified[replace_indices] = features[replace_indices]

    modified_pred = model(baseline_modified)
    _, modified_predicted_class = torch.max(modified_pred[0], dim=1)

    print("\n========== PREDICTION FOR MODIFIED BASELINE (0.05% real features) ==========")
    print(f"> Replaced indices: {len(replace_indices.tolist())}")
    print(f"> Modified predicted class: {modified_predicted_class.item()}")
    print(f"> Modified logits: {modified_pred[0].detach().cpu().numpy()}")

    cosine_similarity_modified = torch.nn.functional.cosine_similarity(features, baseline_modified, dim=1)
    print(f"> Cosine similarity (modified vs. features): {cosine_similarity_modified.mean().item():.4f}")

    print("========== COMPUTE IG METHODS ==========")
    ig_methods = ['ig', 'random']
    
    # Define top-k values
    max_k = min(num_patches, 1000)  # Cap at 1000 or num_patches
    tumor_small_k = np.unique(np.logspace(np.log10(1), np.log10(max_k * 0.1), num=10, dtype=int))  # Small k for tumor
    tumor_large_k = np.array([int(max_k * p) for p in [0.2, 0.4, 0.6, 0.8, 0.95]])  # Larger k
    normal_large_k = np.unique(np.logspace(np.log10(max_k * 0.5), np.log10(max_k), num=10, dtype=int))  # Large k for normal
    normal_small_k = np.array([1, 5, 10, 20, 50])  # Small k
    top_k_values = tumor_small_k + tumor_large_k if pred_class == 1 else normal_small_k + normal_large_k
    top_k_values = np.sort(np.unique(np.concatenate([tumor_small_k, tumor_large_k, normal_small_k, normal_large_k])))  # Symmetric
    print(f"\n> Top-k values (Class {'Tumor' if pred_class == 1 else 'Normal'}): {top_k_values}")
    
    random_mask = generate_random_mask(num_patches, fraction=0.0)  # Disable random mask
    print(f"\n> Number of patches: {num_patches}")
    print(f"> Number of masked patches: {random_mask.sum()}")

    results_all = {}
    saliency_maps = {}
    
    for ig_name in ig_methods:
        print(f"\n>> Running IG method: {ig_name}")
        if ig_name == 'random':
            saliency_map = np.random.rand(num_patches)
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
                "x_steps": 50,
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
                top_k_values, 0, model, args.device,
                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                keep_monotonous=False
            )
            aic_score = compute_pic_metric(
                features.cpu().numpy(), saliency_map, random_mask,
                top_k_values, 1, model, args.device,
                baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                keep_monotonous=False
            )
            results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
            print(f"  - SIC AUC: {sic_score.auc:.6f}\n  - AIC AUC: {aic_score.auc:.6f}")
        except Exception as e:
            print(f"  > Failed to compute PIC for {ig_name}: {e}")
            results_all[ig_name] = None

    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        if v:
            print(f"{k.upper():<5} : SIC = {v['SIC']:.6f} | AIC = {v['AIC']:.6f}")
        else:
            print(f"{k.upper():<5} : FAILED")

def main(args):
    basenames = ['test_001']
    for basename in basenames:
        print(f"\n=== Processing slide: {basename} ===")
        compute_one_slide(args, basename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()
    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)
    
    args = parse_args_from_config(config)
    args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    main(args)