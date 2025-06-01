import os
import sys
import argparse
import torch
import yaml
import numpy as np

# Add CLAM and attribution paths
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("attr_method"))



from clam import load_clam_model

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
    # === CONFIG FOR CLAM MODEL === 
    
    def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        
        
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3

        inputs = inputs.to(device).clone().detach().requires_grad_(True)
        if was_batched:
            inputs = inputs.squeeze(0)  # [1, N, D] -> [N, D]

        model.eval()
        outputs = model(inputs, [inputs.shape[0]])
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
            class_idx = call_model_args.get("target_class_idx", 0)
            target = logits[:, class_idx]  # [N]
            grads = torch.autograd.grad(
                outputs=target,
                inputs=inputs,
                grad_outputs=torch.ones_like(target),
                retain_graph=False,
                create_graph=False
            )[0]
            grads_np = grads.detach().cpu().numpy()
            if was_batched or grads_np.ndim == 2:
                grads_np = np.expand_dims(grads_np, axis=0)  # Ensure [1, N, D]
            return {INPUT_OUTPUT_GRADIENTS: grads_np}

        return logits 
 
    ig = AttrMethod()    
    return ig, call_model_function


def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def main(args, config):
    # === Setup fixed paths ===
    # feature_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/data_for_checking/clam_camelyon16/tumor_028.pt"
    # checkpoint_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/checkpoints_simea/clam/camelyon16/s_1_checkpoint.pt"
    # memmap_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/memmap_path"
    
    basename = "tumor_028"
    fold_idx = 1
 
    feature_path = os.path.join(config['paths']['feature_files'], f"{basename}.pt")
    checkpoint_path = os.path.join(config['paths'][f'for_ig_checkpoint_path_fold_{fold_idx}'])
    mammap_path = os.path.join(config['paths']['memmap_path'])
    os.makedirs(mammap_path, exist_ok=True)

    
    # === Create dummy args and override from config ===
    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size

    # === Load CLAM model ===
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)


     # === Load IG module config ===
    ig_module, call_model_function  = load_ig_module(args) 
     
    # =======Loop thru each example and compute Load feature file ===
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)

    if features.dim() == 3:
        features = features.squeeze(0)

    print(f"> Feature shape: {features.shape}")


    # === Predict class ===
    with torch.no_grad():
        output = model(features, [features.shape[0]])
        logits, probs, predicted_class, *_ = output

    pred_class = predicted_class.item()
    print(f"\n> Prediction Complete")
    print(f"  - Logits         : {logits}")
    print(f"  - Probabilities  : {probs}")
    print(f"  - Predicted class: {pred_class}")
    print(f"\n> Running Integrated Gradients for class {pred_class}")


    # === Generate average baseline ===
    mean_vector = features.mean(dim=0, keepdim=True)     # shape: [1, D]
    baseline = mean_vector.expand_as(features)           # shape: [N, D]
    print(f"> Baseline shape   : {baseline.shape}")
    
    # === Add batch dimension ===
    features = features.unsqueeze(0)  # [1, N, D]
    mean_vector = features.mean(dim=1, keepdim=True)         # [1, 1, D]
    baseline = mean_vector.expand_as(features)               # [1, N, D]
    print(f"> Feature shape  : {features.shape}")
    print(f"> Baseline shape : {baseline.shape}")
        
    # === Run Integrated Gradients ===
    kwargs = {
        "x_value": features,
        "call_model_function": call_model_function,
        "model": model,
        "baseline_features": baseline,
        "memmap_path": memmap_path,
        "x_steps": 50,
        "device": args.device,
        "call_model_args": {"target_class_idx": pred_class}, 
        "batch_size": 500 # Optional, can be adjusted
    }

    attribution_values = ig_module.GetMask(**kwargs)
    saliency_map = attribution_values.mean(1)

    print(f"  - Attribution shape: {attribution_values.shape}")
    print(f"  - Mean score shape : {scores.shape}")
    # print(f"  - Top scores       : {scores.topk(5).values.cpu().numpy()}")

    # === Save results ===
    
    # === Compute Performance Information Curve (PIC) ===
    sys.path.append(os.path.join("src/evaluation"))
    from pic import compute_pic_metric, generate_random_mask, PicMetricResultBasic, ComputePicMetricError 

    num_patches = features.shape[1]  # Number of patches N
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

        print(f"\n> PIC Metrics Computed")
        print(f"  - SIC AUC: {sic_score.auc:.3f}")
        print(f"  - AIC AUC: {aic_score.auc:.3f}")

        # === Save results ===
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

    # === Load YAML config ===
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)
    # args.device = "cpu"
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # args.ig_name = 'ig' 
    # args.ig_name = 'cig'
    args.ig_name = 'idg' 
    

     
    print("=== Configuration Loaded ===")
    print(f"> Device       : {args.device}")
    print(f"> Dropout      : {args.drop_out}")
    print(f"> Embed dim    : {args.embed_dim}")
    print(f"> Model type   : {args.model_type}")
    main(args, config)
