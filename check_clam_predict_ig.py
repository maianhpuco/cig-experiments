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
from attr_method_old.integrated_gradient import IntegratedGradients  # Only using IG here
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS 

def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def main(args):
    # === Setup fixed paths ===
    feature_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/data_for_checking/clam_camelyon16/tumor_028.pt"
    checkpoint_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/checkpoints_simea/clam/camelyon16/s_1_checkpoint.pt"
    memmap_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/memmap_path"
    os.makedirs(memmap_path, exist_ok=True)

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

    # === Load feature file ===
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
    
    ig = IntegratedGradients()
    print(f"\n> Running Integrated Gradients for class {pred_class}")
    
    def call_model_function(features, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        features = features.to(device)
        features.requires_grad_(True)
        model.eval()

        was_batched = features.dim() == 3
        if was_batched:
            features = features.squeeze(0)  # [1, N, D] -> [N, D]
            
        model_output = model(features, [features.shape[0]])
        logits = model_output[0] if isinstance(model_output, tuple) else model_output

        target_class_idx = call_model_args['target_class_idx']
        target_logit = logits[:, target_class_idx]  # shape: [N] — no .sum() here!
        print(f">>>>>>> Target logit shape: {target_logit.shape}")  # should be [N]
        grads = torch.autograd.grad(
            outputs=target_logit,
            inputs=features,
            grad_outputs=torch.ones_like(target_logit),
            create_graph=False,
            retain_graph=False
        )[0]

        gradients = grads.detach().cpu().numpy()
        if was_batched:
            gradients = np.expand_dims(gradients, axis=0)  # shape: [1, N, D] 
        print(f">>>>>>> Gradients shape: {gradients.shape}")  # should be [N, D]
        return {INPUT_OUTPUT_GRADIENTS: gradients}
    
 
    kwargs = {
        "x_value": features,
        "call_model_function": call_model_function,
        "model": model,
        "baseline_features": baseline,
        "memmap_path": memmap_path,
        "x_steps": 50,
        "device": args.device,
        "call_model_args": {"target_class_idx": pred_class}
    }

    attribution_values = ig.GetMask(**kwargs)
    scores = attribution_values.mean(1)

    print(f"  - Attribution shape: {attribution_values.shape}")
    print(f"  - Mean score shape : {scores.shape}")
    print(f"  - Top scores       : {scores.topk(5).values.cpu().numpy()}")

    # === Save results ===

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

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.ig_name = 'integrated_gradient'

    print("=== Configuration Loaded ===")
    print(f"> Device       : {args.device}")
    print(f"> Dropout      : {args.drop_out}")
    print(f"> Embed dim    : {args.embed_dim}")
    print(f"> Model type   : {args.model_type}")
    main(args)
