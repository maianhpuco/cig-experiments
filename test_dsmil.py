import os
import yaml
import argparse
import torch
import sys
import numpy as np
import pandas as pd
from torch.nn.functional import sigmoid
from sklearn.metrics import accuracy_score, roc_auc_score

# Add model paths
sys.path.append(os.path.abspath("src/models"))
sys.path.append(os.path.abspath("src/models/classifiers"))
sys.path.append(os.path.join("src/externals/dsmil-wsi"))

from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga
import dsmil as mil

def load_dsmil_model(args, ckpt_path):
    print(f"[INFO] Loading DSMIL model from: {ckpt_path}")

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                   dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    model = mil.MILNet(i_classifier, b_classifier).cuda()

    try:
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] DSMIL model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        raise e

    model.eval()
    return model

def predict(model, test_dataset, device, dataset_name, dropout_patch=0.0):
    model.eval()
    all_preds, all_labels, all_slide_ids = [], [], []
    all_logits, all_probs = [], []
    all_feature_counts = []
    
    for idx, data in enumerate(test_dataset):
        if dataset_name == 'camelyon16':
            (features, label) = data
        elif dataset_name in ['tcga_renal', 'tcga_lung']:
            (features, label, _) = data 
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        slide_id = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"\n[INFO] Processing {idx+1}/{len(test_dataset)}: {slide_id}")

        features = features.to(device)
        if dropout_patch > 0:
            features = dropout_patches(features, 1 - dropout_patch)
        features = features.view(-1, args.feats_size)

        with torch.no_grad():
            ins_pred, bag_pred, A, _ = model(features)
            logits = bag_pred.squeeze()
            probs = sigmoid(logits)  # Use sigmoid for binary classification
            pred = (probs[1] > 0.5).long().item()  # Threshold at 0.5 for class 1

        print(f"   - Prediction: {pred} | Ground Truth: {label} | Probabilities: {probs.cpu().numpy()}")

        all_preds.append(pred)
        all_labels.append(label)
        all_slide_ids.append(slide_id)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_feature_counts.append(features.shape[0])

    # Debug class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"[INFO] Test set label distribution: {dict(zip(unique_labels, counts))}")

    return all_preds, all_labels, all_slide_ids, all_logits, all_probs, all_feature_counts

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    return feats[random_indices]

def main(args):
    fold_id = args.fold
    device = args.device

    model = load_dsmil_model(args, args.ckpt_path)
    model.to(device)

    if args.dataset_name == 'camelyon16':
        label_dict = {'normal': 0, 'tumor': 1}
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        
        train_dataset, val_dataset, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['data_dir'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        print(f"[INFO] Train Set Size: {len(train_dataset)}")
        print(f"[INFO] Val Set Size: {len(val_dataset)}")
        print(f"[INFO] Test Set Size: {len(test_dataset)}")
    else:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        print(f"[INFO] Using label_dict: {label_dict}")
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

    print("========= Start Prediction on Test Set ===========")
    all_preds, all_labels, all_slide_ids, all_logits, all_probs, all_feature_counts = predict(
        model=model,
        test_dataset=test_dataset,
        device=device,
        dataset_name=args.dataset_name,
        dropout_patch=args.dropout_patch if hasattr(args, 'dropout_patch') else 0.0
    )

    os.makedirs(args.paths['predictions_dir'], exist_ok=True)
    df_dict = {
        'slide_id': all_slide_ids,
        'feature_count': all_feature_counts,
        'true_label': all_labels,
        'pred_label': all_preds,
        'logits': [logit.tolist() for logit in all_logits],
        'probs': [prob.tolist() for prob in all_probs],
    }

    num_classes = len(all_probs[0])
    for c in range(num_classes):
        df_dict[f'logit_class_{c}'] = [logit[c] for logit in all_logits]
        df_dict[f'prob_class_{c}'] = [prob[c] for prob in all_probs]

    output_df = pd.DataFrame(df_dict)
    save_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    output_df.to_csv(save_path, index=False)
    print(f"[INFO] Predictions saved to {save_path}")

    print("========= Compute Metrics ===========")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Accuracy: {accuracy:.4f}")

    try:
        if num_classes == 2:
            auc_score = roc_auc_score(all_labels, [p[1] for p in all_probs])
            print(f"[INFO] AUC: {auc_score:.4f}")
        else:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            print(f"[INFO] AUC (ovr): {auc_score:.4f}")
    except Exception as e:
        auc_score = 'N/A'
        print(f"[WARNING] Could not compute AUC: {e}")

    metrics_df = pd.DataFrame([{
        'fold': fold_id,
        'accuracy': round(accuracy, 4),
        'auc': round(auc_score, 4) if auc_score != 'N/A' else 'N/A'
    }])

    save_path = os.path.join(args.paths['metrics_dir'], f'acc_auc_fold{fold_id}.csv')
    metrics_df.to_csv(save_path, index=False)
    print(f"[INFO] Metrics saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with DSMIL model')
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--ig_name', type=str, default='integrated_gradient')
    parser.add_argument('--fold', type=int, default=1, help='Fold number to evaluate')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to model checkpoint')

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

    if args.ckpt_path is None:
        args.ckpt_path = args.paths.get(f'for_ig_checkpoint_path_fold_{args.fold}', None)
        if args.ckpt_path is None:
            raise ValueError(f"[ERROR] No checkpoint path provided and no 'for_ig_checkpoint_path_fold_{args.fold}' found in config")
        print(f"[INFO] Using checkpoint from config: {args.ckpt_path}")
    else:
        print(f"[INFO] Using checkpoint from argument: {args.ckpt_path}")

    main(args)