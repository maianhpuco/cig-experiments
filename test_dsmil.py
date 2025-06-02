import os
import yaml
import argparse
import torch
import sys
import numpy as np
import pandas as pd

# Add model paths
ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(ig_path)
sys.path.append(clf_path)
 
from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga
sys.path.append(os.path.join("src/externals/dsmil-wsi"))
import dsmil as mil 

def load_dsmil_model(args, ckpt_path):

 
    """
    Load a trained DSMIL model from a checkpoint file.

    Args:
        args: argparse.Namespace containing model arguments like feats_size, num_classes, etc.
        ckpt_path (str): Path to the saved .pth model checkpoint

    Returns:
        model (MILNet): Loaded DSMIL model in evaluation mode
    """

    print(f"[INFO] Loading DSMIL model from: {ckpt_path}")

    # Initialize instance and bag classifiers
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                               dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()

    # Wrap into MILNet
    model = mil.MILNet(i_classifier, b_classifier).cuda()

    # Load checkpoint weights
    state_dict = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    print("[INFO] DSMIL model loaded and set to eval mode.")
    return model 
def main(args):
    fold_id = args.fold
    device = args.device

    print(f"[INFO] Loading model checkpoint from: {args.ckpt_path}")
# model = load_clam_model(args, args.paths[f'for_ig_checkpoint_path_fold_{fold_id}'], device)
    model = load_dsmil_model(args, args.ckpt_path)
    model.to(device)   
    model.eval()

    print("========= Reading on Test Set ===========")

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

    print("========= Start Prediction on Test Set ===========")
    all_preds, all_labels, all_slide_ids = [], [], []
    all_logits, all_probs = [], []

    # for i in range(len(test_dataset)):
    for idx, data in enumerate(test_dataset):
        if args.dataset_name == 'camelyon16':
            (features, label) = data
        if args.dataset_name in ['tcga_renal', 'tcga_lung']:
            (features, label, _) = data
        basename = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"\nProcessing file {idx + 1}/{len(test_dataset)}: {basename}")
        features = features.to(device) 
        with torch.no_grad():
            output = model(features, [features.shape[0]])
            logits, Y_prob, Y_hat, _, _ = output  # Unpack CLAM output tuple
            pred = torch.argmax(logits, dim=1)[0].item()  # Get predicted class
        print(f"Predicted class for {basename} {pred} ground truth {label}")  

        all_preds.append(pred)
        all_labels.append(label)
        all_slide_ids.append(basename)
        all_logits.append(logits.cpu().numpy().flatten())
        all_probs.append(Y_prob.cpu().numpy().flatten())

    # Create prediction folder if not exists
    os.makedirs(args.paths['predictions_dir'], exist_ok=True)

    # Save predictions
    df_dict = {
        'slide_id': all_slide_ids,
        'true_label': all_labels,
        'pred_label': all_preds,
        'logits': [logit.tolist() for logit in all_logits],
        'probs': [prob.tolist() for prob in all_probs],
    }

    # Add logit and prob for each class (multi-class support)
    num_classes = len(all_probs[0])
    for c in range(num_classes):
        df_dict[f'logit_class_{c}'] = [logit[c] for logit in all_logits]
        df_dict[f'prob_class_{c}'] = [prob[c] for prob in all_probs]

    output_df = pd.DataFrame(df_dict)
    save_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    output_df.to_csv(save_path, index=False)
    print(f"[INFO] Predictions saved to {save_path}")
    
    # Compute accuracy after all predictions 
    
    print("========= Compute Accuracy after finish ===========") 
    from sklearn.metrics import accuracy_score, roc_auc_score
 
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Accuracy: {accuracy:.4f}")

    # Compute AUC (only if binary or one-vs-rest for multi-class)
    try:
        if len(set(all_labels)) == 2:
            # Binary case: use prob for class 1
            auc_score = roc_auc_score(all_labels, [p[1] for p in all_probs])
        else:
            # Multi-class: use one-vs-rest mode
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"[INFO] AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"[WARNING] Could not compute AUC: {e}") 
# Save accuracy and AUC to CSV
    metrics_df = pd.DataFrame([{
        'fold': fold_id,
        'accuracy': round(accuracy, 4),
        'auc': round(auc_score, 4) if 'auc_score' in locals() else 'N/A'
    }])

    save_path = os.path.join(args.paths['predictions_dir'], f'acc_auc_{fold_id}.csv')
    metrics_df.to_csv(save_path, index=False)
    print(f"[INFO] Accuracy and AUC saved to {save_path}") 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
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
        args.ckpt_path = args.paths[f'for_ig_checkpoint_path_fold_{args.fold}']
        print(f"[INFO] Using checkpoint from config file: {args.ckpt_path}") 
        
    print(" > Start compute PREDICTION for dataset: ", args.dataset_name)
    main(args)
