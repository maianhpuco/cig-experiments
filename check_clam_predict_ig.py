import os
import sys
import argparse
import torch
import yaml

# Add CLAM model path
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))

from clam import load_clam_model


def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    return parser.parse_args(args=[])


def main(args):
    # Fixed paths for feature and checkpoint
    feature_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/data_for_checking/clam_camelyon16/tumor_028.pt"
    checkpoint_path = "/project/hnguyen2/mvu9/processing_datasets/cig_data/checkpoints_simea/clam/camelyon16/s_1_checkpoint.pt"

    # Create dummy args and override with loaded config
    args_clam = get_dummy_args()
    args_clam.drop_out = args.drop_out
    args_clam.n_classes = args.n_classes
    args_clam.embed_dim = args.embed_dim
    args_clam.model_type = args.model_type
    args_clam.model_size = args.model_size

    # Load CLAM model
    print(f"\n> Loading CLAM model from: {checkpoint_path}")
    model = load_clam_model(args_clam, checkpoint_path, device=args.device)

    # Load feature file
    print(f"\n> Loading feature from: {feature_path}")
    data = torch.load(feature_path)
    features = data['features'] if isinstance(data, dict) else data
    features = features.to(args.device, dtype=torch.float32)

    if features.dim() == 3:  # [1, N, D]
        features = features.squeeze(0)

    print(f"> Feature shape: {features.shape}")

    # Run prediction
    with torch.no_grad():
        output = model(features, [features.shape[0]])
        logits, Y_prob, Y_hat = output

    print(f"\n> Prediction Complete")
    print(f"  - Logits         : {logits}")
    print(f"  - Probabilities  : {Y_prob}")
    print(f"  - Predicted class: {Y_hat.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply config to args
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

    main(args)
