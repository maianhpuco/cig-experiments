'''
this code is to test ig's model on tcga-renal + clam
'''
import os
import sys 
import argparse
import time
import numpy as np
import torch
import yaml
import shutil
from torch import nn

ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)  

from clam import load_clam_model  
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS 
from src.datasets.classification.tcga import return_splits_custom  as return_splits_tcga 
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16

class ModelWrapper:
    """Wraps a model to standardize forward calls for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
            from attr_method.ig import IG as AttrMethod
            print("Using Integrated Gradients (IG) method")
        elif name == 'g':
            from attr_method.g import VanillaGradients as AttrMethod
            print("Using Integrated Gradients (IG) method")
        elif name == 'cig':
            from attr_method.cig import CIG as AttrMethod
            print("Using Cumulative Integrated Gradients (CIG) method")
        elif name == 'idg':
            from attr_method.idg_w_batch import IDG as AttrMethod
            print("Using Integrated Decision Gradients (IDG) method with batch support")
        elif name == 'eg':
            from attr_method.eg import EG as AttrMethod
            print("Using Expected Gradients (EG) method")
        else:
            raise ValueError(f"Unsupported IG method: {name}")
        return AttrMethod()

    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3
        inputs = inputs.to(device)
        
        if not inputs.requires_grad:
            inputs.requires_grad_(True) 
            
        # inputs = inputs.to(device).clone().detach().requires_grad_(True)
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
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}

        _, _, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['pt_files'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        print(f"[INFO] Test Set Size: {len(test_dataset)}")

    elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        split_folder = args.paths['split_folder']
        data_dir_map = args.paths['data_dir']

        train_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
        val_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
        test_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

        train_dataset, val_dataset, test_dataset = return_splits_tcga(
            train_csv_path,
            val_csv_path,
            test_csv_path,
            data_dir_map=data_dir_map,
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        print(f"[INFO] FOLD {fold_id} -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
 
    return test_dataset 
            
     

def get_baseline_features(fold_id, basename, features_size):
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt") 
    baseline_path = os.path.join(args.paths[f'baseline_dir_fold_{fold_id}'], f"{basename}.pt")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline file not found at: {baseline_path}")
    baseline = torch.load(baseline_path).to(args.device, dtype=torch.float32)
    baseline = baseline.squeeze(0) if baseline.dim() == 3 else baseline

    # Resample baseline to match number of patches
    if baseline.shape[0] != features_size:
        print(f"[WARN] Baseline patch count ({baseline.shape[0]}) doesn't match features ({features_size}). Resampling baseline.")
        indices = torch.randint(0, baseline.shape[0], (features_size,), device=baseline.device)
        baseline = baseline[indices]  
    if baseline.shape[0] <100:
        print(f"[WARN] Baseline patch count ({baseline.shape[0]}) is less than 100. This may affect results.") 
    return baseline


def main(args):
    ig_module, call_model_function = load_ig_module(args) 
    memmap_path = os.path.join(args.paths['memmap_path'], f'{args.ig_name}')
    
    if os.path.exists(memmap_path):
        shutil.rmtree(memmap_path)
    os.makedirs(memmap_path, exist_ok=True)
    
    # checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{args.fold}']) 
    checkpoint_path = args.ckpt_path  
    
    print("--------num classes", args.n_classes)
    model = load_clam_model(args, checkpoint_path, device=args.device) 
     
     
    print("================= LOADING DATASET and COMPUTE IG =================")   
    test_dataset = load_dataset(args, fold_id=1)  # Assuming fold_id is 1 for this example 
    for idx, data in enumerate(test_dataset):
        if args.dataset_name == 'camelyon16':
            (features, label) = data
        if args.dataset_name in ['tcga_renal', 'tcga_lung']:
            (features, label, _) = data
            
        features = features.unsqueeze(0) if features.dim() == 2 else features    
        basename = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"\nProcessing file {idx + 1}/{len(test_dataset)}: {basename}")
        print(f"  >  Features shape: {features.shape}")
        features = features.to(args.device, dtype=torch.float32)        
        with torch.no_grad():
            model_wrapper = ModelWrapper(model, model_type='clam')
            logits = model_wrapper.forward(features)
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_class = torch.max(logits, dim=1)
        pred_class = predicted_class.item()
        print(f"Predicted class: {pred_class}")

        baseline  =  get_baseline_features(fold_id, basename, features.shape[-2]).to(args.device, dtype=torch.float32)
        print(f"  >  Baseline shape: {baseline.shape}")     
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
        return 
        save_dir = os.path.join(
            args.paths['attribution_scores_folder'], f'{args.ig_name}', f'fold_{args.fold}') 
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{basename}.npy") 


        if isinstance(attribution_values, torch.Tensor):
            attribution_values = attribution_values.detach().cpu().numpy()
        print(f"- Score shape: {attribution_values.shape}")
        np.save(save_path, attribution_values)
                
        from utils_plot import min_max_scale  # if not already imported
        scores = np.mean(np.abs(attribution_values), axis=-1).squeeze()
        normalized_scores = min_max_scale(scores.copy())
                
        print("=====Sanity Check the result======= ")
        print(f"  >  Shape          : {normalized_scores.shape}")
        print(f"  >  First 3 values : {[float(f'{s:.6f}') for s in normalized_scores[:3]]}")
        print(f"  >  Sum            : {np.sum(normalized_scores):.6f}")
        print(f"  >  Min value      : {np.min(normalized_scores):.6f}")
        print(f"  >  Max value      : {np.max(normalized_scores):.6f}")
        print(f"  >  Non-zero count : {np.count_nonzero(normalized_scores)} / {len(normalized_scores)}")
        print(f"save scores to {save_path}, basename: {basename}, fold_id: {fold_id}, ig_name: {args.ig_name}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--fold_start', type=int, default=1, help='Fold index to evaluate')
    parser.add_argument('--fold_end', type=int, default=1, help='Fold index to evaluate')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run the model on')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Optional checkpoint path override')

    args = parser.parse_args()

    with open(f'{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    if args.dataset_name =='tcga_renal':
        args.data_dir_map = config['paths']['data_dir'] 
    
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = 'cpu'
    os.makedirs(args.paths['attribution_scores_folder'], exist_ok=True)

    print(" > Start compute IG for dataset: ", args.dataset_name)
    for fold_id in range(args.fold_start, args.fold_end + 1):  # Assuming folds are 1 to 5
        args.fold = fold_id # Assuming fold_id is 1 for this example
        main(args)
