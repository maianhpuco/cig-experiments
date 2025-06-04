import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import sys

# Add model paths
sys.path.append(os.path.join("src/models"))
sys.path.append(os.path.join("src/models/classifiers"))

from mlp_trainer import load_model_mlp
# from src.datasets.load_dataset import load_dataset  # <-- Your external dataset loader

from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import softmax

def load_dataset(args):
    fold_id = args.fold 
    
    from src.datasets.classification.tcga import return_splits_custom  as return_splits_tcga 
    from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16  
    
    if args.dataset_name == 'camelyon16':
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        label_dict = {'normal': 0, 'tumor': 1}

        train_dataset, val_dataset, test_dataset = return_splits_camelyon16(
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
       
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print(f"[INFO] FOLD {fold_id} -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}") 
    
    return train_dataset, val_dataset, test_dataset 



def main(args):
    fold_id = args.fold
    device = args.device

    print(f"[INFO] Loading model checkpoint from: {args.ckpt_path}")
    model, device = load_model_mlp(args, args.ckpt_path)
    model.eval()

    print("========= Loading Dataset ==========")
    _, _, test_dataset = load_dataset(args)

    all_preds, all_labels, all_slide_ids = [], [], []
    all_logits, all_probs = [], []
    all_feature_counts = []

    print("========= Start Prediction on Test Set ==========")
    for idx, data in enumerate(test_dataset):
        if args.dataset_name == 'camelyon16':
            features, label = data
        elif args.dataset_name in ['tcga_renal', 'tcga_lung']:
            features, label, _ = data
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")

        basename = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"[{idx + 1}/{len(test_dataset)}] Slide: {basename}")

        all_slide_ids.append(basename)
        all_labels.append(label)
        all_feature_counts.append(features.shape[0])

        features = features.to(device)
        with torch.no_grad():
            logits = model(features.unsqueeze(0))
            probs = softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1).item()
        print(f"  → Predicted: {pred}, True: {label}")

        all_preds.append(pred)
        all_logits.append(logits.cpu().numpy().flatten())
        all_probs.append(probs.cpu().numpy().flatten())

    # Save predictions
    os.makedirs(args.paths['predictions_dir'], exist_ok=True)

    df_dict = {
        'slide_id': all_slide_ids,
        'feature_count': all_feature_counts,
        'true_label': all_labels,
        'pred_label': all_preds,
        'logits': [logit.tolist() for logit in all_logits],
        'probs': [prob.tolist() for prob in all_probs]
    }

    num_classes = len(all_probs[0])
    for c in range(num_classes):
        df_dict[f'logit_class_{c}'] = [logit[c] for logit in all_logits]
        df_dict[f'prob_class_{c}'] = [prob[c] for prob in all_probs]

    output_df = pd.DataFrame(df_dict)
    pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    output_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved to: {pred_path}")

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"[WARNING] Could not compute AUC: {e}")
        auc = 'N/A'

    print(f"[METRIC] Accuracy: {accuracy:.4f}")
    print(f"[METRIC] AUC: {auc:.4f}" if auc != 'N/A' else "[WARNING] AUC not available")

    metrics_df = pd.DataFrame([{
        'fold': fold_id,
        'accuracy': round(accuracy, 4),
        'auc': round(auc, 4) if auc != 'N/A' else 'N/A'
    }])
    metrics_path = os.path.join(args.paths['predictions_dir'], f'acc_auc_{fold_id}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Accuracy and AUC saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--fold', type=int, default=1, help='Fold index to evaluate')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run the model on')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Optional checkpoint path override')
 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)

    args.dataset_name = config['dataset_name']
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.paths['predictions_dir'], exist_ok=True)
    if args.ckpt_path is not None:
        args.ckpt_path = args.ckpt_path
        print(f"[INFO] Using checkpoint from --ckpt_path argument: {args.ckpt_path}")
    else:
        print("PLEAES INPUT CHECKPOINTS IN ARGS") 
        # args.ckpt_path = args.paths[f'for_ig_checkpoint_path_fold_{args.fold}']
        # print(f"[INFO] Using checkpoint from config file: {args.ckpt_path}") 
        
    print(" > Start compute PREDICTION for dataset: ", args.dataset_name)
    main(args) 