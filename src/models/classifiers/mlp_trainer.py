from __future__ import print_function

import argparse
import os
import math
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from datetime import datetime
# Add model paths
# ig_path = os.path.abspath(os.path.join("src/models"))
clf_path = os.path.abspath(os.path.join("src/models/classifiers"))
# sys.path.append(ig_path)
sys.path.append(clf_path)

# Assuming the model classes (Bag_Classifier, Attn_Net, Attn_Net_Gated) are in a file called `models.py`
from mlp_classifier import MLP_Classifier

def load_model_mlp(args, checkpoint_path=None):
    """
    Load the Bag_Classifier model for training.
    
    Args:
        args: ArgumentParser object containing model configuration (e.g., model_size, drop_out, etc.)
        checkpoint_path (str, optional): Path to a pre-trained model checkpoint.
    
    Returns:
        model: Initialized Bag_Classifier model, optionally with loaded weights.
        device: Device (CUDA or CPU) where the model is loaded.
    """
    device = args.device
    
    # Initialize model
    model = MLP_Classifier(
        gate=args.gate,  # Whether to use gated attention
        size_arg=args.model_size,  # 'small' or 'big'
        dropout=args.drop_out,  # Dropout rate
        n_classes=args.n_classes,  # Number of classes
        subtyping=args.subtyping,  # Subtyping flag
        embed_dim=args.embed_dim  # Input feature dimension
    )
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    
    # Move model to device
    model = model.to(device)
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model, device
 
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
    best_model_path = os.path.join(args.results_dir, f'best_model.pth')

    print(f"[INFO] Starting training for Fold {args.fold} on device {device} ...")
    
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()

        total_batches = len(train_loader)
        log_interval = max(1, total_batches // 10)  # Print every ~10% of loader

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
                percent = (batch_idx + 1) / total_batches * 100
                print(f"  [Epoch {epoch+1}] {batch_idx+1}/{total_batches} ({percent:.1f}%) - Loss: {loss.item():.4f}")

        train_loss /= total_batches

        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time

        print(f"[Epoch {epoch+1}/{args.max_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} "
              f"| Time: {epoch_duration:.2f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'epoch': epoch
            }, best_model_path)
            print(f"[INFO] ✔ Best model updated and saved at: {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience counter: {patience_counter}/{patience}")
            if args.early_stopping and patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break
    
    # Test evaluation
    print("[INFO] Evaluating best model on test set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"[RESULT] Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Accuracy: {test_acc:.4f}")
    
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

def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Bag_Classifier model
        loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation
    
    Returns:
        loss: Average loss
        auc: AUC score
        acc: Accuracy
    """
    model.eval()
    loss = 0.0
    preds = []
    labels = []
    
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            loss += criterion(logits, label).item()
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    loss /= len(loader)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    
    # Compute AUC and accuracy
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(labels, preds[:, 1]) if len(np.unique(labels)) > 1 else 0.0
    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    
    return loss, auc, acc


