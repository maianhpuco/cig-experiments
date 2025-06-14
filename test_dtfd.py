import os
import yaml
import argparse
import torch
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Add model paths
sys.path.append(os.path.abspath("src/models"))
sys.path.append(os.path.abspath("src/models/classifiers"))
sys.path.append(os.path.join("src/externals/dtfd-mil")) 

from src.datasets.classification.camelyon16 import return_splits_custom as return_splits_camelyon16
from src.datasets.classification.tcga import return_splits_custom as return_splits_tcga

from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from Model.network import Classifier_1fc, DimReduction


def load_dtfd_model(ckpt_path, device, args):
    """Load the DTFD model components from checkpoint."""
    classifier = Classifier_1fc(args.mDim, args.num_cls, args.droprate).to(device)
    attention = Attention(args.mDim).to(device)
    dimReduction = DimReduction(1024, args.mDim, numLayer_Res=args.numLayer_Res).to(device)
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

    all_preds, all_labels, all_slide_ids, all_logits, all_probs, all_feature_counts = [], [], [], [], [], []
    for idx, data in enumerate(test_dataset):
        if dataset_name == 'camelyon16':
            features, label = data
        else:
            features, label, _ = data

        slide_id = test_dataset.slide_data['slide_id'].iloc[idx]
        print(f"[INFO] Processing {idx+1}/{len(test_dataset)}: {slide_id}")

        with torch.no_grad():
            features = features.to(device)
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
        all_feature_counts.append(features.shape[0])
    return all_preds, all_labels, all_slide_ids, all_logits, all_probs, all_feature_counts


def main(args):
    device = args.device
    fold_id = args.fold

    model = load_dtfd_model(args.ckpt_path, device, args)

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
    else:
        label_dict = args.label_dict if hasattr(args, "label_dict") else None
        train_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'train.csv')
        val_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'val.csv')
        test_csv_path = os.path.join(args.paths['split_folder'], f'fold_{fold_id}', 'test.csv')

        train_dataset, val_dataset, test_dataset = return_splits_tcga(
            train_csv_path, val_csv_path, test_csv_path,
            data_dir_map=args.paths['data_dir'],
            label_dict=label_dict,
            seed=42,
            print_info=True
        )

    print("========= Start Prediction on Test Set ===========")
    all_preds, all_labels, all_slide_ids, all_logits, all_probs, all_feature_counts = predict(
        model=model,
        test_dataset=test_dataset,
        device=device,
        dataset_name=args.dataset_name
    )

    os.makedirs(args.paths['predictions_dir'], exist_ok=True)
    results_dict = {
        'slide_id': all_slide_ids,
        'feature_count': all_feature_counts,
        'true_label': all_labels,
        'pred_label': all_preds,
        'logits': [logit.tolist() for logit in all_logits],
        'probs': [prob.tolist() for prob in all_probs],
    }

    for c in range(len(all_probs[0])):
        results_dict[f'logit_class_{c}'] = [logit[c] for logit in all_logits]
        results_dict[f'prob_class_{c}'] = [prob[c] for prob in all_probs]

    df_out = pd.DataFrame(results_dict)
    csv_pred_path = os.path.join(args.paths['predictions_dir'], f'test_preds_fold{fold_id}.csv')
    df_out.to_csv(csv_pred_path, index=False)
    print(f"[INFO] Predictions saved to {csv_pred_path}")

    # Accuracy & AUC
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
    print("Unique predicted labels:", df_out['pred_label'].unique())

    df_metrics = pd.DataFrame([{
        'fold': fold_id,
        'accuracy': round(accuracy, 4),
        'auc': round(auc_score, 4) if auc_score != 'N/A' else 'N/A'
    }])
    csv_metric_path = os.path.join(args.paths['predictions_dir'], f'acc_auc_{fold_id}.csv')
    df_metrics.to_csv(csv_metric_path, index=False)
    print(f"[INFO] Accuracy and AUC saved to {csv_metric_path}")


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
    args.device = "cpu"
    if args.ckpt_path is None:
        args.ckpt_path = args.paths[f'for_ig_checkpoint_path_fold_{args.fold}']
        print(f"[INFO] Using checkpoint from config: {args.ckpt_path}")
    else:
        print(f"[INFO] Using checkpoint from argument: {args.ckpt_path}")

    main(args)
