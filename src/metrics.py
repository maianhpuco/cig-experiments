import numpy as np
import torch
from torch.nn.functional import softmax
from typing import Callable, Tuple, List

def compute_aic_and_sic(
    model: Callable,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    attribution_map: torch.Tensor,
    target_class: int,
    call_model_function: Callable,
    steps: int = 100
) -> Tuple[float, float]:
    """
    Compute AIC (Accuracy Info Curve) and SIC (Softmax Info Curve).
    """
    flat_attr = attribution_map.flatten()
    sorted_indices = torch.argsort(flat_attr, descending=True)
    total_points = flat_attr.numel()
    step_size = max(1, total_points // steps)

    aic_scores = []
    sic_scores = []

    with torch.no_grad():
        original_logits = call_model_function(model, input_tensor.unsqueeze(0), target_class_idx=target_class)
        original_pred = torch.argmax(original_logits, dim=1).item()

    for i in range(0, total_points, step_size):
        k = min(i + step_size, total_points)
        mask = torch.ones_like(flat_attr)
        mask[sorted_indices[:k]] = 0
        mask = mask.view(attribution_map.shape)

        perturbed = input_tensor * mask + baseline_tensor * (1 - mask)

        with torch.no_grad():
            logits = call_model_function(model, perturbed.unsqueeze(0), target_class_idx=target_class)
            prob = softmax(logits, dim=1)[0, target_class].item()
            pred = torch.argmax(logits, dim=1).item()

        aic_scores.append(1.0 if pred == original_pred else 0.0)
        sic_scores.append(prob)

    dx = 1.0 / steps
    return np.trapz(aic_scores, dx=dx), np.trapz(sic_scores, dx=dx)

def compute_insertion_auc(
    model: Callable,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    attribution_map: torch.Tensor,
    target_class: int,
    call_model_function: Callable,
    steps: int = 100
) -> float:
    flat_attr = attribution_map.flatten()
    sorted_indices = torch.argsort(flat_attr, descending=True)
    total_points = flat_attr.numel()
    step_size = max(1, total_points // steps)

    insertion_scores = []

    for i in range(0, total_points, step_size):
        k = min(i + step_size, total_points)
        mask = torch.zeros_like(flat_attr)
        mask[sorted_indices[:k]] = 1
        mask = mask.view(attribution_map.shape)

        inserted = input_tensor * mask + baseline_tensor * (1 - mask)
        with torch.no_grad():
            logits = call_model_function(model, inserted.unsqueeze(0), target_class_idx=target_class)
            prob = softmax(logits, dim=1)[0, target_class].item()
            insertion_scores.append(prob)

    return np.trapz(insertion_scores, dx=1.0 / steps)

def compute_deletion_auc(
    model: Callable,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    attribution_map: torch.Tensor,
    target_class: int,
    call_model_function: Callable,
    steps: int = 100
) -> float:
    flat_attr = attribution_map.flatten()
    sorted_indices = torch.argsort(flat_attr, descending=True)
    total_points = flat_attr.numel()
    step_size = max(1, total_points // steps)

    deletion_scores = []

    for i in range(0, total_points, step_size):
        k = min(i + step_size, total_points)
        mask = torch.ones_like(flat_attr)
        mask[sorted_indices[:k]] = 0
        mask = mask.view(attribution_map.shape)

        deleted = input_tensor * mask + baseline_tensor * (1 - mask)
        with torch.no_grad():
            logits = call_model_function(model, deleted.unsqueeze(0), target_class_idx=target_class)
            prob = softmax(logits, dim=1)[0, target_class].item()
            deletion_scores.append(prob)

    return np.trapz(deletion_scores, dx=1.0 / steps)

def rank_methods(
    results: List[Tuple[str, float, float, float, float]]
) -> List[Tuple[str, float, float, float, float]]:
    """
    Rank attribution methods by AIC (↑), SIC (↑), Insertion AUC (↑), Deletion AUC (↓).
    """
    method_ranks = {m[0]: 0 for m in results}

    def assign_ranks(metric_index: int, reverse: bool):
        sorted_methods = sorted(results, key=lambda x: x[metric_index], reverse=reverse)
        for rank, (name, *_vals) in enumerate(sorted_methods, 1):
            method_ranks[name] += rank

    assign_ranks(1, True)   # AIC ↑
    assign_ranks(2, True)   # SIC ↑
    assign_ranks(3, True)   # Insertion AUC ↑
    assign_ranks(4, False)  # Deletion AUC ↓

    return sorted(results, key=lambda x: method_ranks[x[0]] / 4.0)
