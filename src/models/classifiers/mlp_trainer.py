from __future__ import print_function

import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score

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
        subtyping=getattr(args, 'subtyping', False),
        embed_dim=args.embed_dim
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
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
            data, label = batch[:2]  # support (data, label) or (data, label, metadata)
            data, label = data.to(device), label.to(device).long()
            logits = model(data)
            loss += criterion(logits, label).item()

            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            labels.append(label.cpu().numpy())

    loss /= len(loader)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    if preds.shape[1] == 2:
        auc = roc_auc_score(labels, preds[:, 1])
    else:
        auc = roc_auc_score(labels, preds, multi_class='ovr')

    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    return loss, auc, acc

def train(checkpoint_dir, datasets, args):
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
    best_model_path = os.path.join(args.results_dir, 'best_model.pth')

    print(f"[INFO] Starting training for Fold {args.fold} on device {device} ...")

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        log_interval = max(1, total_batches // 10)

        for batch_idx, batch in enumerate(train_loader):
            data, label = batch[:2]
            data, label = data.to(device), label.to(device).long()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                percent = (batch_idx + 1) / total_batches * 100
                print(f"  [Epoch {epoch+1}] {batch_idx+1}/{total_batches} ({percent:.1f}%) - Loss: {loss.item():.4f}")

        train_loss /= total_batches
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{args.max_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

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
            if getattr(args, 'early_stopping', False) and patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

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
