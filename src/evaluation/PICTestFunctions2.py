import numpy as np
import torch
import os
import glob
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
from scipy import interpolate

class ModelWrapper:
    """Wraps a model to standardize forward calls for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            input = input.squeeze(0)
        if self.model_type == 'clam':
            output = self.model(input, [input.shape[0]])
            logits = output[0] if isinstance(output, tuple) else output
        else:
            logits = self.model(input)
        return logits

def create_neutral_features(full_features: np.ndarray, patch_mask: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    neutral_features = full_features.copy()
    neutral_features[~patch_mask] = baseline[~patch_mask] if baseline.shape == full_features.shape else baseline
    return neutral_features

def generate_random_mask(num_patches: int, fraction: float = 0.01) -> np.ndarray:
    mask = np.zeros(num_patches, dtype=bool)
    indices = np.random.choice(num_patches, size=int(num_patches * fraction), replace=False)
    mask[indices] = True
    return mask

def estimate_feature_information(features: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """
    Estimate information content as the cosine similarity to a reference (usually the original features).
    
    Args:
        features (np.ndarray): Neutral or masked features [N, D].
        reference (np.ndarray, optional): Original full features [N, D]. If None, falls back to norm-based method.
    
    Returns:
        float: Mean similarity score as proxy for retained information.
    """
    if reference is None:
        print("Warning: No reference provided for cosine similarity. Using L2 norm as fallback.")
    norm_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    norm_ref = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.sum(norm_features * norm_ref, axis=1)
    return cosine_sim.mean()

class ComputePicMetricError(Exception):
    pass

def getPrediction(input: torch.Tensor, model_wrapper: ModelWrapper, intendedClass: int, method: int, device: str) -> Tuple[float, int]:
    input = input.to(device)
    logits = model_wrapper.forward(input)
    print(f"> Logits for method={'SIC' if method == 0 else 'AIC'}: {logits.detach().cpu().numpy()}", end=' ')
    if intendedClass == -1:
        _, index = torch.max(logits, dim=1)
        softmax = torch.nn.functional.softmax(logits, dim=1)[0, index[0]].detach().cpu().numpy()
        return softmax, index[0].item()
    else:
        if method == 0:  # SIC
            softmax = torch.nn.functional.softmax(logits, dim=1)[0, intendedClass].detach().cpu().numpy()
            return softmax, -1
        elif method == 1:  # AIC
            _, index = torch.max(logits, dim=1)
            return 1.0 if index[0].item() == intendedClass else 0.0, -1

class PicMetricResultBasic(NamedTuple):
    curve_x: Sequence[float]
    curve_y: Sequence[float]
    auc: float

def compute_pic_metric(features: np.ndarray, saliency_map: np.ndarray, random_mask: np.ndarray, 
                      top_k_values: Sequence[int], method: int, model, device: str,
                      baseline: np.ndarray, min_pred_value: float = 0.3, 
                      keep_monotonous: bool = False, num_data_points: int = 1000) -> PicMetricResultBasic:
    """
    Computes Performance Information Curve (SIC or AIC) for a single WSI feature set using top-k patches.
    
    Args:
        features: Feature tensor of shape [N, D] (e.g., D=1024).
        saliency_map: Saliency scores of shape [N,].
        random_mask: Binary mask of shape [N,].
        top_k_values: List of k values for top-k patch selection.
        method: 0 for SIC (remove top-k), 1 for AIC (add top-k).
        model: CLAM model (wrapped in ModelWrapper internally).
        device: Device for computation.
        baseline: Baseline feature tensor of shape [N, D].
        min_pred_value: Minimum prediction confidence for original features.
        keep_monotonous: Whether to enforce monotonicity in the curve.
        num_data_points: Number of data points for the interpolated curve.
    
    Returns:
        PicMetricResultBasic containing the curve and AUC.
    """
    model_wrapper = ModelWrapper(model, model_type='clam')
    neutral_features = []
    predictions = []
    entropy_pred_tuples = []
    num_patches = features.shape[0]

    if baseline.shape != features.shape:
        raise ValueError(f"Baseline shape {baseline.shape} must match features shape {features.shape}")

    original_features_info = estimate_feature_information(features, reference=features)
    fully_neutral_features = create_neutral_features(features, random_mask, baseline)
    fully_neutral_info = estimate_feature_information(fully_neutral_features, reference=features)

    input_features = torch.from_numpy(features).unsqueeze(0).to(device)
    original_pred, correct_class = getPrediction(input_features, model_wrapper, -1, method, device)
    print(f"\n{'SIC' if method == 0 else 'AIC'} - Original prediction: {original_pred:.6f} (Class: {correct_class})")

    if original_pred < min_pred_value:
        raise ComputePicMetricError(f"Original prediction {original_pred} is below min_pred_value {min_pred_value}")

    fully_neutral_pred_features = torch.from_numpy(fully_neutral_features).unsqueeze(0).to(device)
    fully_neutral_pred, _ = getPrediction(fully_neutral_pred_features, model_wrapper, correct_class, method, device)
    print(f"{'SIC' if method == 0 else 'AIC'} - Fully neutral prediction: {fully_neutral_pred:.6f}")

    neutral_features.append(fully_neutral_features)
    predictions.append(fully_neutral_pred)

    max_normalized_pred = 0.0

    sorted_indices = np.argsort(-saliency_map)  # Descending order for top-k selection

    for k in top_k_values:
        if k > num_patches:
            print(f"Skipping k={k} as it exceeds number of patches ({num_patches})")
            continue

        if method == 0:  # SIC: Remove top-k patches
            patch_mask = np.ones(num_patches, dtype=bool)
            patch_mask[sorted_indices[:k]] = False  # Remove top-k
            visible_fraction = (num_patches - k) / num_patches
        else:  # Method 1: AIC: Add top-k patches
            patch_mask = np.zeros(num_patches, dtype=bool)
            patch_mask[sorted_indices[:k]] = True  # Add top-k
            visible_fraction = k / num_patches

        if random_mask.sum() > 0:
            patch_mask = np.logical_or(patch_mask, random_mask)
        else:
            visible_fraction = patch_mask.sum() / num_patches
        print(f">> Top-k {k}/{num_patches} ({visible_fraction*100:.2f}% {'visible'removed' if method == 0 else 'added'})")

        neutral_features_current = create_neutral_features(features, patch_mask, baseline)

        info = estimate_feature_information(neutral_features_current, reference=features)
        pred_input = torch.from_numpy(neutral_features_current).unsqueeze(0).to(device)
        pred, _ = getPrediction(pred_input, model_wrapper, correct_class, method, device)
        normalized_info = np.linspace(0, 1, len(info))
        # normalized_info = (info - fully_neutral_info) / (original_features_info - fully_neutral_info)
        # normalized_info = np.clip(normalized_info, 0.0, 1.0)
        # normalized_pred = (pred - fully_neutral_pred) / (original_pred - fully_neutral_pred) if (original_pred - fully_neutral_pred) > 1e-6 else pred
        # normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        # max_normalized_pred = max(max_normalized_pred, normalized_pred)

        print(f"{'SIC' if method == 0 else 'AIC'} - Top-k {k}: Info: {info:.5f}, Pred = {pred:.5f} | Normed Info = {normalized_info:.4f}, Normed Pred = {normalized_pred:.4f}")
        entropy_pred_tuples.append((normalized_info, max_normalized_pred))
        
        # else:
        #     entropy_pred_tuples.append((normalized_info, normalized_pred))

        neutral_features.append(neutral_features_current)
        predictions.append(pred)

    entropy_pred_tuples.append((0.0, 0.0))
    entropy_pred_tuples.append((1.0, 1.0))

    sorted_tuples = sorted(entropy_pred_tuples, key=lambda x: x[0])
    info_data, pred_data = zip(*sorted_tuples)

    interp_func = interpolate.interp1d(x=info_data, y=pred_data, fill_value=(0.0, 1.0), bounds_error=False)

    curve_x = np.linspace(0.0, 1.0, num_data_points)
    curve_y = np.asarray([interp_func(x) for x in curve_x])

    auc = np.trapz(curve_y, curve_x)

    return PicMetricResultBasic(curve_x=curve_x, curve_y=curve_y, auc=auc)