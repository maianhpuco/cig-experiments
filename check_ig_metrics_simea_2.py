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

def compute_one_slide(args, basename, model):

    # basename = 'test_003'
    fold_id = 1
    
    feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
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
      
    print(f"> Loaded baseline from: {baseline_path} | Shape: {baseline.shape}")

    baseline = baseline.squeeze(0) if baseline.dim() == 3 else baseline
    baseline_pred = model(baseline)
    _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1) 
    print(f"> Baseline predicted class: {baseline_predicted_class.item()}")
    print(f"> Baseline logits: {baseline_pred[0].detach().cpu().numpy()}")

    print("========== COMPUTE IG METHODS ==========")
    ig_methods = ['ig', 'cig', 'idg', 'eg', 'random']  # List of IG methods to evmethod
    ig_methods = ['idg']
    tumor_low = np.logspace(np.log10(0.00001), np.log10(0.05), num=7)
    mid = np.linspace(0.2, 0.8, num=3) 
    # # Normal class: more stable, only changes with full signal (thresholds near 1)
    normal_high = 1 - tumor_low[::-1]  # Flip to go toward 1
    # mid = np.linspace(0.1, 0.9, num=10) 
    saliency_thresholds = np.sort(np.unique(np.concatenate([mid, normal_high])))
    top_k = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 30, 50, 100])  # Top-k thresholds for evaluation

    
    random_mask = generate_random_mask(features.shape[0], fraction=0.0)  # Disable random mask
    print(f"\n> Number of patches: {features.shape[0]}")
    print(f"> Number of masked patches: {random_mask.sum()}")

    results_all = {}
    saliency_maps = {}
    # Get labels from pred_df
    
    slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
    pred_label = slide_row['pred_label'].iloc[0] if not slide_row.empty else pred_class
    true_label = slide_row['true_label'].iloc[0] if 'true_label' in slide_row.columns else -1
    results = []
    for ig_name in ig_methods:
        print(f"\n>> Running IG method: {ig_name}")
        if ig_name == 'random':
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
                "x_steps": 50,  # Increase for better IG accuracy
                "device": args.device,
                "call_model_args": {"target_class_idx": pred_class},
                "batch_size": 500
            }
            # try:

            attribution_values = ig_module.GetMask(**kwargs)
            saliency_map = np.abs(np.mean(np.abs(attribution_values), axis=-1)).squeeze()
            saliency_map = saliency_map / (saliency_map.max() + 1e-8)
            print(f"  - Saliency map shape: {saliency_map.shape} Saliency stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")
            # except Exception as e:
            #     print(f"  > Failed for {ig_name}: {e}")
            #     results.append({
            #         "slide_id": basename,
            #         "pred_label": pred_label,
            #         "true_label": true_label,
            #         "IG": ig_name,
            #         "AIC": None,
            #         "SIC": None
            #     })
            #     continue


        sic_score = compute_pic_metric(top_k, 
            features.cpu().numpy(), saliency_map, random_mask,
            saliency_thresholds, 0, model, args.device,
            baseline=baseline.cpu().numpy(), min_pred_value=0.3,
            keep_monotonous=False
        )
        aic_score = compute_pic_metric(top_k,
            features.cpu().numpy(),saliency_map, random_mask,
            saliency_thresholds, 1, model, args.device,
            baseline=baseline.cpu().numpy(), min_pred_value=0.3,
            keep_monotonous=False
        )

        results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
        print(f"  - SIC AUC: {sic_score.auc:.5f}\n  - AIC AUC: {aic_score.auc:.5f}")
        results.append({
                "slide_id": basename,
                "pred_label": pred_label,
                "true_label": true_label,
                "baseline_pred_label": baseline_predicted_class.item(),
                "saliency_map_mean": saliency_map.mean(), 
                "saliency_map_std": saliency_map.std(), 
                "IG": ig_name,
                "AIC": aic_score.auc,
                "SIC": sic_score.auc
            }) 

    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        if v:
            print(f"{k.upper():<5} : SIC = {v['SIC']:.6f} | AIC = {v['AIC']:.6f}")
        else:
            print(f"{k.upper():<5} : FAILED")
            
    return results 
    # output_file = os.path.join(memmap_path, "pic_result03ml")
    # print(f"> Results saved to: {output_file}")im
    
    
def main(args):
    import time 
    import pandas as pd
    fold_id = 1 
    args.fold = 1
    checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'])

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

    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    pred_df = pd.read_csv(pred_path)
    print(f"[INFO] Loaded predictions from fold {args.fold}: {pred_df.shape[0]} samples") 
    tumor_df = pred_df[pred_df['pred_label'] == 1]
    
    basenames = tumor_df['slide_id'].unique().tolist()
    # basenames = ['test_001', 'test_002', 'test_004', 'test_008']
    args.pred_df = tumor_df 
    all_results = [] 
    start = time.time()
    basenames = ['test_069']
    count_total= len(basenames)
    for basename in basenames:
        print(f"\n=== Processing slide: {basename}, {len(all_results) + 1}/{count_total} ===")
        all_results.append(compute_one_slide(args, basename, model))
    
    results_df = pd.DataFrame(all_results)
    output_dir =  os.path.join(args.paths['metrics_dir'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pic_results_fold{fold_id}.csv")

    
    results_df.to_csv(output_path, index=False)
    print(f"\n> Results saved to: {output_path}") 
    avg_results = results_df.groupby("IG")[["AIC", "SIC"]].mean().reset_index()

    print("\n=== Average AIC and SIC per IG Method ===")
    print(avg_results.to_string(index=False)) 
    print(f"\n> Total time taken: {time.time() - start:.2f} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args_from_config(config)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
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
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from PICTestFunctions import compute_pic_metric, generate_random_mask, ModelWrapper

def load_ig_module(args):
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
            print("Using Expected Gradients (EG) method")
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
    args.device = getattr(args, 'device', "cuda" if torch.cuda.is_available() else "cpu")
    return args

def compute_one_slide(args, basename, model):
    fold_id = args.fold
    feature_path = os.path.join(args.paths['feature_files'], f"{basename}.pt")
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

    # Resample baseline to match number of patches
    if baseline.shape[0] != features.shape[0]:
        print(f"[WARN] Baseline patch count ({baseline.shape[0]}) doesn't match features ({features.shape[0]}). Resampling baseline.")
        indices = torch.randint(0, baseline.shape[0], (features.shape[0],), device=baseline.device)
        baseline = baseline[indices]

    baseline_pred = model(baseline)
    _, baseline_predicted_class = torch.max(baseline_pred[0], dim=1)
    print(f"> Baseline predicted class: {baseline_predicted_class.item()}")
    print(f"> Baseline logits: {baseline_pred[0].detach().cpu().numpy()}")

    print("========== COMPUTE IG METHODS ==========")
    ig_methods = ['idg']
    tumor_low = np.logspace(np.log10(0.00001), np.log10(0.05), num=7)
    mid = np.linspace(0.2, 0.8, num=3)
    normal_high = 1 - tumor_low[::-1]
    saliency_thresholds = np.sort(np.unique(np.concatenate([mid, normal_high])))
    top_k = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 30, 50, 100])

    random_mask = generate_random_mask(features.shape[0], fraction=0.0)
    print(f"\n> Number of patches: {features.shape[0]}")
    print(f"> Number of masked patches: {random_mask.sum()}")

    results_all = {}
    results = []
    slide_row = args.pred_df[args.pred_df['slide_id'] == basename]
    pred_label = slide_row['pred_label'].iloc[0] if not slide_row.empty else pred_class
    true_label = slide_row['true_label'].iloc[0] if 'true_label' in slide_row.columns else -1

    for ig_name in ig_methods:
        print(f"\n>> Running IG method: {ig_name}")
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
        attribution_values = ig_module.GetMask(**kwargs)
        saliency_map = np.abs(np.mean(np.abs(attribution_values), axis=-1)).squeeze()
        saliency_map = saliency_map / (saliency_map.max() + 1e-8)

        print(f"  - Saliency map shape: {saliency_map.shape} Stats: mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")

        sic_score = compute_pic_metric(top_k, features.cpu().numpy(), saliency_map, random_mask,
                                       saliency_thresholds, 0, model, args.device,
                                       baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                                       keep_monotonous=False)
        aic_score = compute_pic_metric(top_k, features.cpu().numpy(), saliency_map, random_mask,
                                       saliency_thresholds, 1, model, args.device,
                                       baseline=baseline.cpu().numpy(), min_pred_value=0.3,
                                       keep_monotonous=False)

        results_all[ig_name] = {"SIC": sic_score.auc, "AIC": aic_score.auc}
        results.append({
            "slide_id": basename,
            "pred_label": pred_label,
            "true_label": true_label,
            "baseline_pred_label": baseline_predicted_class.item(),
            "saliency_map_mean": saliency_map.mean(),
            "saliency_map_std": saliency_map.std(),
            "IG": ig_name,
            "AIC": aic_score.auc,
            "SIC": sic_score.auc
        })

    print("\n=== Summary of PIC Scores ===")
    for k, v in results_all.items():
        print(f"{k.upper():<5} : SIC = {v['SIC']:.6f} | AIC = {v['AIC']:.6f}")

    return results

def main(args):
    fold_id = args.fold = 1
    checkpoint_path = os.path.join(args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'])

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
    basenames = ['test_069']

    print(f"[INFO] Loaded {len(tumor_df)} tumor slides from predictions")

    all_results = []
    start = time.time()
    for idx, basename in enumerate(basenames):
        print(f"\n=== Processing slide: {basename} ({idx + 1}/{len(basenames)}) ===")
        all_results.append(compute_one_slide(args, basename, model))

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(args.paths['metrics_dir'], f"pic_results_fold{fold_id}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n> Results saved to: {output_path}")
    avg_results = results_df.groupby("IG")[["AIC", "SIC"]].mean().reset_index()
    print("\n=== Average AIC and SIC per IG Method ===")
    print(avg_results.to_string(index=False))
    print(f"\n> Total time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args_from_config(config)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
