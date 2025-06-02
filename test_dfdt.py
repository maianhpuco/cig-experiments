import os
import yaml
import argparse
import torch
import sys
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

# Add model paths
sys.path.append(os.path.abspath("src/models"))
sys.path.append(os.path.abspath("src/models/classifiers"))

from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga

sys.path.append(os.path.join("src/externals/dtfd-mil")) 
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction 

def load_dtfd_model(ckpt_path, device, args):
    """Load the DTFD model components from checkpoint."""
    in_chn = 1024  # Fixed input dim
    classifier = Classifier_1fc(args.mDim, args.num_cls, args.droprate).to(device)
    attention = Attention(args.mDim).to(device)
    dimReduction = DimReduction(in_chn, args.mDim, numLayer_Res=args.numLayer_Res).to(device)
    attCls = Attention_with_Classifier(L=args.mDim, num_cls=args.num_cls, droprate=args.droprate_2).to(device)

    if args.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)

    ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(ckpt["classifier"])
    attention.load_state_dict(ckpt["attention"])
    dimReduction.load_state_dict(ckpt["dim_reduction"])
    attCls.load_state_dict(ckpt["att_classifier"])

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()

    return classifier, dimReduction, attention, attCls 

def predict(model, test_dataset, device, dataset_name):
    """Run prediction using the loaded DTFD model on a test set."""
    classifier, dimReduction, attention, attCls = model
    classifier.eval()
    dimReduction.eval()
    attention.eval()
    attCls.eval()

    all_preds, all_labels, all_slide_ids, all_logits, all_probs = [], [], [], [], []
    for idx, data in enumerate(test_dataset): 
        if dataset_name == 'camelyon16':
            features, label = data
        else:
            features, label, _ = data 
        slide_id = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"\n[INFO] Processing {idx+1}/{len(test_dataset)}: {slide_id}") 
        
        with torch.no_grad():
            midFeat = dimReduction(features)
            A = attention(midFeat, isNorm=False).squeeze(0)
            A = torch.softmax(A, dim=0)
            weighted_feat = torch.einsum("ns,n->ns", midFeat, A)
            slide_feat = torch.sum(weighted_feat, dim=0).unsqueeze(0)
            slide_logits = classifier(slide_feat)
            prob = torch.softmax(slide_logits, dim=1)

            pred_label = torch.argmax(prob, dim=1).item()

        all_slide_ids.append(slide_id)
        all_labels.append(label)
        all_preds.append(pred_label)
        all_logits.append(slide_logits.squeeze().cpu())
        all_probs.append(prob.squeeze().cpu())

    return all_preds, all_labels, all_slide_ids, all_logits, all_probs 

def main(args):
    fold_id = args.fold
    device = args.device

    model = load_dsmil_model(args, args.ckpt_path)
    model.to(device)

    if args.dataset_name == 'camelyon16':
        label_dict = {'normal': 0, 'tumor': 1}
        split_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}.csv')
        # df = pd.read_csv(split_csv_path)
        
        # train_count = df['train'].notna().sum()
        # val_count = df['val'].notna().sum()
        # test_count = df['test'].notna().sum()

        # print(f"[INFO] Number of slides in each split:")
        # print(f"  Train: {train_count}")
        # print(f"  Val:   {val_count}")
        # print(f"  Test:  {test_count}")
    
        train_dataset, val_dataset, test_dataset = return_splits_camelyon16(
            csv_path=split_csv_path,
            data_dir=args.paths['data_dir'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )
        print(f"[INFO] Val Set Size: {len(val_dataset)}")
        print(f"[INFO] Test Set Size: {len(test_dataset)}")
        print(f"[INFO] Train Set Size: {len(train_dataset)}")
    else:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        split_folder = args.paths['split_folder']
        train_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'train.csv')
        val_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'val.csv')
        test_csv_path = os.path.join(split_folder, f'fold_{fold_id}', 'test.csv')

        train_dataset, val_dataset, test_dataset = return_splits_tcga(
            train_csv_path, val_csv_path, test_csv_path,
            data_dir_map=args.paths['data_dir'], label_dict=label_dict, seed=42, print_info=True
        )
        print(f"[INFO] Val Set Size: {len(val_dataset)}")
        print(f"[INFO] Test Set Size: {len(test_dataset)}")
        print(f"[INFO] Train Set Size: {len(train_dataset)}")
        return 
    print("========= Start Prediction on Test Set ===========")
    all_preds, all_labels, all_slide_ids, all_logits, all_probs = predict(
        model=model,
        test_dataset=test_dataset,
        device=device,
        dataset_name=args.dataset_name
    )

    os.makedirs(args.paths['predictions_dir'], exist_ok=True)
    df_dict = {
        'slide_id': all_slide_ids,
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

    print("========= Compute Accuracy after finish ===========")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Accuracy: {accuracy:.4f}")

    try:
        if len(set(all_labels)) == 2:
            auc_score = roc_auc_score(all_labels, [p[1] for p in all_probs])
        else:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"[INFO] AUC: {auc_score:.4f}")
    except Exception as e:
        auc_score = 'N/A'
        print(f"[WARNING] Could not compute AUC: {e}")

    metrics_df = pd.DataFrame([{
        'fold': fold_id,
        'accuracy': round(accuracy, 4),
        'auc': round(auc_score, 4) if auc_score != 'N/A' else 'N/A'
    }])

    save_path = os.path.join(args.paths['predictions_dir'], f'acc_auc_{fold_id}.csv')
    metrics_df.to_csv(save_path, index=False)
    print(f"[INFO] Accuracy and AUC saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config', default='clam_camelyon16.yaml')
    parser.add_argument('--ig_name', default='integrated_gradient')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--ckpt_path', type=str, default=None)

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
        args.ckpt_path = args.paths[f'for_ig_checkpoint_path_fold_{args.fold}']
        print(f"[INFO] Using checkpoint from config: {args.ckpt_path}")
    else:
        print(f"[INFO] Using checkpoint from argument: {args.ckpt_path}")

    main(args)
