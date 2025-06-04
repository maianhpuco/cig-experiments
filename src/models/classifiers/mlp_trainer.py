from __future__ import print_function

import os
import sys
import time
import math
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

# Add model paths
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
sys.path.append(clf_path)

from mlp_classifier import MLP_Classifier  # your MLP model

def load_model_mlp(args, checkpoint_path=None):
    device = args.device

    model = MLP_Classifier(
        gate=args.gate,
        size_arg=args.model_size,
        dropout=args.drop_out,
        n_classes=args.n_classes,
        subtyping=args.subtyping,
        embed_dim=args.embed_dim
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    return model, device


def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                data, label = batch
            elif len(batch) == 3:
                data, label, _ = batch
            else:
                raise ValueError("Unexpected batch format")

            data, label = data.to(device), label.to(device)
            logits = model(data)
            loss += criterion(logits, label).item()

            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            labels.append(label.cpu().numpy())

    loss /= len(loader)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(labels, preds[:, 1]) if len(np.unique(labels)) > 1 else 0.0
    acc = accuracy_score(labels, np.argmax(preds, axis=1))

    return loss, auc, acc


def train(checkpoint_dir, datasets, args):
    """
    Train the MLP_Classifier model for one fold.

    Args:
        checkpoint_dir: Directory to save checkpoints
        datasets: Tuple of (train_dataset, val_dataset, test_dataset)
        args: argparse.Namespace with training configuration

    Returns:
        results: Dictionary with train/val/test metrics
        test_auc: AUC on test set
        val_auc: AUC on validation set
        test_acc: Accuracy on test set
        val_acc: Accuracy on validation set
    """
    train_dataset, val_dataset, test_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model, device = load_model_mlp(args, checkpoint_path=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    patience = getattr(args, 'patience', 20)
    patience_counter = 0
    best_model_path = os.path.join(args.results_dir, f'best_model.pth')

    print(f"[INFO] Starting training for Fold {args.fold} on device {device} ...")

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if len(batch) == 2:
                data, label = batch
            elif len(batch) == 3:
                data, label, _ = batch
            else:
                raise ValueError("Unexpected batch format")

            data, label = data.to(device), label.to(device)
            # print(f"label: {label}, shape: {label.shape}, n_classes: {args.n_classes}") 
            optimizer.zero_grad()
            logits = model(data)
            label = label.long()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Fold {args.fold}, Epoch {epoch+1}/{args.max_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'epoch': epoch
            }, best_model_path)
            print(f"[INFO] ✔ Best model saved at: {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience counter: {patience_counter}/{patience}")
            if args.early_stopping and patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    # Test
    print("[INFO] Loading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"[RESULT] Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Accuracy: {test_acc:.4f}")

    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_acc': val_acc,
        'test_acc': test_acc
    }

    return results, test_auc, val_auc, test_acc, val_acc
