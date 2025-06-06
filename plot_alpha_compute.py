import os
import sys
import argparse
import time
import numpy as np
import torch
import yaml
import pandas as pd
from tqdm import tqdm

# Extend sys.path to include model directories
sys.path.extend([
    os.path.abspath("src/models"),
    os.path.abspath("src/models/classifiers"),
])

from clam import load_clam_model
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from utils_plot import min_max_scale


class ModelWrapper:
    def __init__(self, model, model_type='clam'):
        self.model = model
        self.model_type = model_type

    def forward(self, input):
        if input.dim() == 3:
            input = input.squeeze(0)
        if self.model_type == 'clam':
            output = self.model(input, [input.shape[0]])
            logits = output[0] if isinstance(output, tuple) else output
        else:
            logits = self.model(input)
        return logits



def load_ig_module(args):
    def get_module_by_name(name):
        print(f"Loading IG method: {name}")
        if name == 'ig':
            from attr_method_plot_alpha.ig import IG as AttrMethod
            print("Using Integrated Gradients (IG) method")
        elif name == 'g':
            from attr_method_plot_alpha.g import VanillaGradients as AttrMethod
            print("Using Integrated Gradients (IG) method")
        elif name == 'cig':
            from attr_method_plot_alpha.cig import CIG as AttrMethod
            print("Using Cumulative Integrated Gradients (CIG) method")
        else:
            raise ValueError(f"Unsupported IG method: {name}")
        return AttrMethod()

    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3
        inputs = inputs.to(device)
        
        if not inputs.requires_grad:
            inputs.requires_grad_(True) 
            
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

def load_dataset(args, fold_id):
    if args.dataset_name == 'camelyon16':
        csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}
        _, _, test_dataset = return_splits_camelyon16(
            csv_path=csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
    else:
        train_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'train.csv')
        val_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'val.csv')
        test_csv = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'test.csv')
        test_dataset = return_splits_tcga(
            train_csv,
            val_csv,
            test_csv,
            data_dir_map=args.paths['data_dir'],
            label_dict=args.label_dict,
            seed=42,
            print_info=True
        )[2]
    return test_dataset


def get_baseline_features(args, fold_id, basename, features_size):
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    baseline = torch.load(baseline_path).to(args.device)
    if baseline.dim() == 3:
        baseline = baseline.squeeze(0)
    if baseline.shape[0] != features_size:
        idx = torch.randint(0, baseline.shape[0], (features_size,))
        baseline = baseline[idx]
    return baseline

def save_stacked_attributions(attributions, save_prefix):
    os.makedirs(save_prefix, exist_ok=True)

    # Get only the alpha_samples (shape [7, N, D])
    if isinstance(attributions, dict) and "alpha_samples" in attributions:
        alpha_samples = attributions["alpha_samples"]  # [7, N, D]
    else:
        raise ValueError("Expected dictionary with 'alpha_samples' key")

    # Compute mean over D: [7, N]
    reduced_attr = np.mean(np.abs(alpha_samples), axis=-1)

    # Check if the reduced attribution is all (or nearly all) zero
    attr_sum = np.sum(reduced_attr)
    if attr_sum < 1e-6:
        raise ValueError(f"[ERROR] Attribution matrix appears to be empty (sum={attr_sum:.4e}). Check the attribution pipeline.")

    # Save as a single (7, N) matrix
    save_path = os.path.join(save_prefix, "attr_alpha_avg.npy")
    np.save(save_path, reduced_attr)
    print(f">>>> SAVED the file with shape {reduced_attr.shape} and sum {attr_sum:.4f} at {save_path}")
    return reduced_attr
 
 
def main(args):
    fold_id = args.fold
    ig_module, call_model_function = load_ig_module(args)
    model = load_clam_model(args, args.ckpt_path, device=args.device)
    test_dataset = load_dataset(args, fold_id=fold_id)

    # Load prediction dataframe
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df[(pred_df['true_label'] == 1) & (pred_df['pred_label'] == 1)]
    slide_ids = set(pred_df['slide_id'].tolist())

    print(f"[INFO] Total slides in test set: {len(test_dataset)}")
    print(f"[INFO] Slides with pred=1 and label=1: {len(slide_ids)}")

    for idx, data in enumerate(test_dataset):
        if args.dataset_name == 'camelyon16':
            features, label = data
        else:
            features, label, _ = data

        features = features.unsqueeze(0) if features.dim() == 2 else features
        features = features.to(args.device)
        basename = test_dataset.slide_data['slide_id'].iloc[idx]

        if basename not in slide_ids:
            print(f"[SKIP] Slide {basename} not matching (label=1, pred=1)")
            continue

        baseline = get_baseline_features(args, fold_id, basename, features.shape[-2])

        kwargs = {
            "x_value": features,
            "call_model_function": call_model_function,
            "model": model,
            "baseline_features": baseline,
            "x_steps": 50,
            "device": args.device,
            "call_model_args": {"target_class_idx": 1},
        }

        attributions = ig_module.GetMask(**kwargs)

        save_prefix = os.path.join(
            args.paths['attr_score_for_multi_alpha_plot_dir'], f"{args.ig_name}", f"fold_{fold_id}", basename
        )
        mean_attr = save_stacked_attributions(attributions, save_prefix)

        scores = np.mean(np.abs(mean_attr), axis=-1)
        normalized = min_max_scale(scores)
        print(f"[SAVE] Slide: {basename} | Attr shape: {normalized.shape} | Sum: {np.sum(normalized):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ig_name', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ckpt_path', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    print(f"[START] Computing {args.ig_name} attributions for fold {args.fold} on {args.dataset_name}")
    main(args)
