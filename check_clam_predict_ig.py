import os
import sys
import argparse
import torch
import yaml

# Add CLAM model path
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))
sys.path.append(os.path.join("attr_method"))  # IG module location

from clam import load_clam_model
from integrated_gradient import IntegratedGradients


def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def call_model_function(x, model, call_model_args=None, expected_keys=None):
    x = x.to(next(model.parameters()).device)
    model.eval()
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        output = model(x, [x.shape[0]])
        logits = output[0]  # output = (logits, probs, predicted_class, ...)
        return logits[:, call_model_args['target_class_idx']]


def main(args):
    
    if args.ig_name == 'integrated_gradient':
        from attr_method.integrated_gradient import IntegratedGradients as AttrMethod
    elif args.ig_name == 'vanilla_gradient':
        from attr_method.vanilla_gradient import VanillaGradients as AttrMethod
    elif args.ig_name == 'contrastive_gradient':
        from attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod
    elif args.ig_name == 'expected_gradient':
        from attr_method.expected_gradient import ExpectedGradients as AttrMethod
    elif args.ig_name == 'integrated_decision_gradient':
        from attr_method.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod
    elif args.ig_name == 'optim_square_integrated_gradient':
        from attr_method.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    elif args.ig_name == 'square_integrated_gradient':
        from attr_method.square_integrated_gradient import SquareIntegratedGradients as AttrMethod
 
    # === Fixed paths ===
    feature_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/data_for_checking/clam_camelyon16/tumor_028.pt"
    checkpoint_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/checkpoints_simea/clam/camelyon16/s_1_checkpoint.pt"

    # === Create dummy args and override ===
    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size

    # === Load model ===
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)

    # === Load features ===
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)

    if features.dim() == 3:
        features = features.squeeze(0)
    print(f"> Feature shape: {features.shape}")

    # === Prediction ===
    with torch.no_grad():
        output = model(features, [features.shape[0]])
        logits, probs, predicted_class, *_ = output

    pred_class = predicted_class.item()
    print(f"\n> Prediction Complete")
    print(f"  - Logits         : {logits}")
    print(f"  - Probabilities  : {probs}")
    print(f"  - Predicted class: {pred_class}")

    # === IG Attribution ===
    print(f"\n> Running Integrated Gradients for class {pred_class}")
    ig = AttrMethod()

    # Baseline: zero vector (shape [N, D])
    baseline = torch.zeros_like(features).to(args.device)

    attribution_values = ig.GetMask(
        x_value=features,
        baseline_features=baseline,
        call_model_function=call_model_function,
        model=model,
        call_model_args={"target_class_idx": pred_class},
        device=args.device,
        x_steps=50
    )

    scores = attribution_values.mean(1)
    print(f"  - Attribution shape: {attribution_values.shape}")
    print(f"  - Mean score shape : {scores.shape}")
    print(f"  - Top scores       : {scores.topk(5).values.cpu().numpy()}")

    print("\n> Done.")


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

    print("=== Configuration Loaded ===")
    print(f"> Dropout      : {args.drop_out}")
    print(f"> Embed dim    : {args.embed_dim}")
    print(f"> Model type   : {args.model_type}")
    print(f"> Device       : {args.device}")
   
    args.ig_name = 'integrated_gradient'
    main(args)
