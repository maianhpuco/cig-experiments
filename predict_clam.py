import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Append CLAM path
sys.path.append(os.path.join("src/externals/CLAM"))

# Model loading
from clam import load_clam_model
from src.datasets.classification.camelyon16 import return_splits_custom as camelyon_splits
from src.datasets.classification.tcga import return_splits_custom as tcga_splits

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def predict_single_fold(args, cfg, fold_id):
    dataset_name = cfg["dataset_name"]
    label_dict_cfg = cfg["label_dict"]
    label_dict = {v: int(k) for k, v in label_dict_cfg.items()}

    split_folder = cfg["paths"]["split_folder"]
    pt_files_dir = cfg["paths"]["pt_files"]
    predictions_root = cfg["paths"]["predictions_dir"]

    pred_save_dir = os.path.join(predictions_root, f"fold_{fold_id}")
    os.makedirs(pred_save_dir, exist_ok=True)

    # Load split
    if dataset_name == "camelyon16":
        csv_path = os.path.join(split_folder, f"fold_{fold_id}.csv")
        _, _, test_dataset = camelyon_splits(
            csv_path=csv_path,
            data_dir=pt_files_dir,
            label_dict=label_dict,
            seed=args.seed,
            print_info=False,
            use_h5=True
        )
    elif dataset_name == "tcga_renal":
        train_csv = os.path.join(split_folder, f"fold_{fold_id}", "train.csv")
        val_csv = os.path.join(split_folder, f"fold_{fold_id}", "val.csv")
        test_csv = os.path.join(split_folder, f"fold_{fold_id}", "test.csv")
        _, _, test_dataset = tcga_splits(
            train_csv, val_csv, test_csv,
            data_dir_map=cfg["paths"]["data_dir"],
            label_dict=label_dict,
            seed=args.seed,
            print_info=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"[Fold {fold_id}] Test samples: {len(test_dataset)}")

    # Load model
    checkpoint_key = f"for_ig_checkpoint_path_fold_{fold_id}"
    ckpt_path = cfg["paths"].get(checkpoint_key)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_clam_model(args, ckpt_path, device=args.device)
    model.eval()

    for idx in tqdm(range(len(test_dataset)), desc=f"[Fold {fold_id}] Predicting"):
        slide_id = test_dataset.slide_data['slide_id'].iloc[idx]
        label = test_dataset.slide_data['label'].iloc[idx]
        feat_path = os.path.join(pt_files_dir, f"{slide_id}.pt")
        features = torch.load(feat_path)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(args.device)

        with torch.no_grad():
            logits, _, _ = model(features, [features.shape[0]])
            pred_class = torch.argmax(logits, dim=1)[0].item()

        save_path = os.path.join(pred_save_dir, f"{slide_id}.npz")
        np.savez(save_path, logits=logits.cpu().numpy(), pred_class=pred_class)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(range(args.fold_start, args.fold_end + 1))
    print(f"Running prediction for folds: {folds}")

    for fold_id in folds:
        predict_single_fold(args, cfg, fold_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLAM Inference Script")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--fold_start', type=int, required=True)
    parser.add_argument('--fold_end', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, default="clam_sb")
    parser.add_argument('--model_size', type=str, default="small")
    parser.add_argument('--drop_out', type=float, default=0.25)
    args = parser.parse_args()

    main(args)
