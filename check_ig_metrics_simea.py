import os
import sys
import argparse
import torch
import yaml
import numpy as np
import random
from scipy.stats import pearsonr

# Add paths for model, attribution, and evaluation code
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("src/attr_method"))
sys.path.append(os.path.join("src/evaluation"))

from clam import load_clam_model
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from PICTestFunctions import compute_pic_metric, generate_random_mask, ComputePicMetricError, ModelWrapper

#  Offline Baseline Pool Creation:

def sample_random_features(dataset, num_files=20):
    indices = np.random.choice(len(dataset), num_files, replace=False)
    feature_list = []
    for idx in indices:
        features, _, _ = dataset[idx]
        features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
        if features.size(0) > 128:
            features = features[:128]
        feature_list.append(features)
    padded = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
    flattened = padded.view(-1, padded.size(-1))
    return flattened

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
            from attr_method.idg_w_batch import IDG as AttrMethod 
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
    fold_id = 1
    feature_path = os.path.join(config['paths']['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(config['paths'][f'for_ig_checkpoint_path_fold_{fold_id}'])
    memmap_path = os.path.join(config['paths']['memmap_path'])
    os.makedirs(memmap_path, exist_ok=True)

    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size
    
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)
    model.eval()

    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)
    if features.dim() == 3:
        features = features.squeeze(0)
    print(f"> Feature shape: {features.shape}")

    with torch.no_grad():
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(features.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, dim=1)
    pred_class = predicted_class.item()
    
    
    #==== sampling baseline features ==== 
    print(f"\n> Prediction Complete\n  - Logits: {logits}\n  - Probabilities: {probs}\n  - Predicted class: {pred_class}")
    from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon 
    split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')

    train_dataset, _, test_dataset = return_splits_camelyon(
        csv_path=split_csv_path,
        data_dir=args.paths['pt_files'],
        label_dict={'normal': 0, 'tumor': 1},
        seed=args.seed,
        print_info=False,
        use_h5=True
    ) 


    num_patches = features.shape[1]

    stacked_features_baseline = sample_random_features(test_dataset).to(args.device, dtype=torch.float32)
    # no need - justtest the baseline pool 
    sampled_indices = np.random.choice(stacked_features_baseline.shape[0], (1, features.shape[1]), replace=True)
    baseline = stacked_features_baseline[sampled_indices].squeeze(0)  # shape: [N, D]
    print(f"> Baseline shape: {baseline.shape}")
    return 
    baseline_pred = model(baseline.squeeze(0))
    print(f"> Baseline prediction: {baseline_pred}")

    ig_methods = ['ig', 'cig', 'idg', 'eg']
    saliency_thresholds = np.linspace(0.005, 0.75, 20)
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
            saliency_map = np.abs(np.mean(np.abs(attribution_values), axis=-1)).squeeze()
            saliency_map = saliency_map / (saliency_map.max() + 1e-8)
            saliency_maps[ig_name] = saliency_map
            print(f"  - Saliency stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")

            sic_score = compute_pic_metric(features.squeeze().cpu().numpy(), saliency_map, random_mask, saliency_thresholds, 0, model, args.device, dataset_mean, dataset_std, min_pred_value=0.3, keep_monotonous=False)
            aic_score = compute_pic_metric(features.squeeze().cpu().numpy(), saliency_map, random_mask, saliency_thresholds, 1, model, args.device, dataset_mean, dataset_std, min_pred_value=0.3, keep_monotonous=False)

            results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
            print(f"  - SIC AUC: {sic_score.auc:.3f}\n  - AIC AUC: {aic_score.auc:.3f}")
        except Exception as e:
            print(f"  > Failed for {ig_name}: {e}")
            results_all[ig_name] = None

    print("\n=== Saliency Map Correlations ===")
    for m1 in ig_methods:
        for m2 in ig_methods:
            if m1 < m2 and m1 in saliency_maps and m2 in saliency_maps:
                corr, _ = pearsonr(saliency_maps[m1], saliency_maps[m2])
                print(f"{m1.upper()} vs {m2.upper()}: Pearson correlation = {corr:.3f}")

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
