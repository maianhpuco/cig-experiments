import os
import sys
import argparse
import time
import numpy as np
import torch
import yaml
import shutil

# Add model paths
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))

from mlp_trainer import load_model_mlp
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from utils_plot import min_max_scale


def load_ig_module(args):
    def get_module_by_name(name):
        print(f"Loading IG method: {name}")
        if name == 'ig':
            from attr_method.ig import IG as AttrMethod
        elif name == 'g':
            from attr_method.g import VanillaGradients as AttrMethod
        elif name == 'cig':
            from attr_method.cig import CIG as AttrMethod
        elif name == 'idg':
            from attr_method.idg_w_batch import IDG as AttrMethod
        elif name == 'eg':
            from attr_method.eg import EG as AttrMethod
        else:
            raise ValueError(f"Unsupported IG method: {name}")
        return AttrMethod()

    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        inputs = inputs.to(device).clone().detach().requires_grad_(True)
        model.eval()
        print("input shape", inputs.shape)
        logits = model(inputs)
        print(logits)
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
            if grads_np.ndim == 2:
                grads_np = np.expand_dims(grads_np, axis=0)
            return {INPUT_OUTPUT_GRADIENTS: grads_np}

        return logits

    return get_module_by_name(args.ig_name), call_model_function


def load_dataset(args, fold_id):
    if args.dataset_name == 'camelyon16':
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}
        _, _, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        train_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'train.csv')
        val_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'val.csv')
        test_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'test.csv')
        _, _, test_dataset = return_splits_tcga(
            train_csv, val_csv, test_csv,
            data_dir_map=args.paths['data_dir'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    return test_dataset


def get_baseline_features(args, fold_id, basename, features_size):
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    baseline = torch.load(baseline_path).to(args.device, dtype=torch.float32)
    baseline = baseline.squeeze(0) if baseline.dim() == 3 else baseline
    if baseline.shape[0] != features_size:
        indices = torch.randint(0, baseline.shape[0], (features_size,), device=baseline.device)
        baseline = baseline[indices]
    return baseline


def main(args):
    ig_module, call_model_function = load_ig_module(args)
    memmap_path = os.path.join(args.paths['memmap_path'], f'{args.ig_name}')
    if os.path.exists(memmap_path):
        shutil.rmtree(memmap_path)
    os.makedirs(memmap_path, exist_ok=True)

    print("Loading MLP model...")
    model, _ = load_model_mlp(args, args.ckpt_path)

    test_dataset = load_dataset(args, fold_id=args.fold)
    for idx, data in enumerate(test_dataset):
        
        features, label = data[:2]
        features = features.unsqueeze(0) if features.dim() == 2 else features
        basename = test_dataset.slide_data['slide_id'].iloc[idx]
        # ---- check if it already exist 
        save_dir = os.path.join(args.paths['attribution_scores_folder'], f'{args.ig_name}', f'fold_{args.fold}') 
        save_path = os.path.join(save_dir, f"{basename}.npy") 
        if args.skip_if_exists and os.path.isfile(save_path):
            print(f"[{idx+1}/{len(test_dataset)}] Skipping {basename} (already exists: {save_path})")
            continue  
        #------ 
        features = features.to(args.device, dtype=torch.float32)

        with torch.no_grad():
            logits = model(features)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        baseline = get_baseline_features(args, args.fold, basename, features.shape[1])

        kwargs = {
            "x_value": features,
            "call_model_function": call_model_function,
            "model": model,
            "baseline_features": baseline,
            "memmap_path": memmap_path,
            "x_steps": 50,
            "device": args.device,
            "call_model_args": {"target_class_idx": pred_class},
            "batch_size": 500
        }

        attribution_values = ig_module.GetMask(**kwargs)
        
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, attribution_values)

        scores = np.mean(np.abs(attribution_values), axis=-1).squeeze()
        norm_scores = min_max_scale(scores.copy())
        print(f"[{idx+1}/{len(test_dataset)}] Saved: {save_path} | sum: {np.sum(norm_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ig_name', required=True)
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=1)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--skip_if_exists', type=int, default=1, help='Skip if attribution file already exists')
 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for k, v in config.items():
        if k == 'paths':
            args.paths = v
        else:
            setattr(args, k, v)

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    for fold_id in range(args.fold_start, args.fold_end + 1):
        args.fold = fold_id
        main(args)

