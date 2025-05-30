import numpy as np
import torch
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
from scipy import interpolate
'''
- create_neutral_features: Replaces create_blurred_image. It uses a binary mask to keep certain patches unchanged and replaces others with the baseline (mean feature vector).
- generate_random_mask: Adapted to create a 1D mask for patches instead of a 2D pixel mask.
- estimate_feature_information: Uses the mean L2 norm of feature vectors instead of WebP compression-based entropy.

Input Handling: Takes feature tensors [N, 512] and saliency scores [N,] instead of images and 2D saliency maps.
Baseline: Uses the mean feature vector as the baseline for neutralization.
'''
def create_neutral_features(full_features: np.ndarray, patch_mask: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Creates a neutralized feature set by replacing unmasked patches with a baseline.
    
    Args:
        full_features: Original feature tensor of shape [N, 512].
        patch_mask: Binary mask of shape [N,], where True indicates patches to keep.
        baseline: Baseline feature vector of shape [512,] or [N, 512].
    
    Returns:
        Neutralized feature tensor of shape [N, 512].
    """
    neutral_features = full_features.copy()
    neutral_features[~patch_mask] = baseline[~patch_mask] if baseline.shape == full_features.shape else baseline
    return neutral_features

def generate_random_mask(num_patches: int, fraction: float = 0.01) -> np.ndarray:
    """
    Generates a random patch mask.
    
    Args:
        num_patches: Number of patches in the WSI.
        fraction: Fraction of patches to set to True.
    
    Returns:
        Binary mask of shape [num_patches,].
    """
    mask = np.zeros(num_patches, dtype=bool)
    indices = np.random.choice(num_patches, size=int(num_patches * fraction), replace=False)
    mask[indices] = True
    return mask

def estimate_feature_information(features: np.ndarray) -> float:
    """
    Estimates the information content of a feature set using the L2 norm.
    
    Args:
        features: Feature tensor of shape [N, 512].
    
    Returns:
        Scalar value representing the information content.
    """
    return np.linalg.norm(features, ord=2, axis=1).mean()

class ComputePicMetricError(Exception):
    pass

def getPrediction(input: torch.Tensor, model, intendedClass: int, method: int, device: str) -> Tuple[float, int]:
    input = input.to(device)
    output = model(input)
    
    if intendedClass == -1:
        _, index = torch.max(output, 1)
        softmax = torch.nn.functional.softmax(output, dim=1)[0, index[0]].detach().cpu().numpy()
        return softmax, index[0]
    else:
        if method == 0:  # SIC
            softmax = torch.nn.functional.softmax(output, dim=1)[0, intendedClass].detach().cpu().numpy()
            return softmax, -1
        elif method == 1:  # AIC
            _, index = torch.max(output, 1)
            return 1.0 if index[0] == intendedClass else 0.0, -1

class PicMetricResultBasic(NamedTuple):
    curve_x: Sequence[float]
    curve_y: Sequence[float]
    auc: float

def compute_pic_metric(features: np.ndarray, saliency_map: np.ndarray, random_mask: np.ndarray, 
                      saliency_thresholds: Sequence[float], method: int, model, device: str,
                      min_pred_value: float = 0.8, keep_monotonous: bool = True, 
                      num_data_points: int = 1000) -> PicMetricResultBasic:
    """
    Computes Performance Information Curve (SIC or AIC) for a single WSI feature set.
    
    Args:
        features: Feature tensor of shape [N, 512].
        saliency_map: Saliency scores of shape [N,].
        random_mask: Binary mask of shape [N,].
        saliency_thresholds: Fractions of important patches to reveal.
        method: 0 for SIC, 1 for AIC.
        model: Model for prediction.
        device: Device for computation.
        min_pred_value: Minimum prediction confidence for original features.
        keep_monotonous: Whether to enforce monotonicity in the curve.
        num_data_points: Number of data points for the interpolated curve.
    
    Returns:
        PicMetricResultBasic containing the curve and AUC.
    """
    neutral_features = []
    predictions = []
    entropy_pred_tuples = []

    # Compute baseline (mean feature vector)
    baseline = np.mean(features, axis=0)  # Shape: [512,]

    # Estimate information content
    original_features_info = estimate_feature_information(features)
    fully_neutral_features = create_neutral_features(features, random_mask, baseline)
    fully_neutral_info = estimate_feature_information(fully_neutral_features)

    # Compute model prediction for original features
    input_features = torch.from_numpy(features).unsqueeze(0)
    original_pred, correctClassIndex = getPrediction(input_features, model, -1, method, device)

    # Compute model prediction for fully neutral features
    fully_neutral_pred_features = torch.from_numpy(fully_neutral_features).unsqueeze(0)
    fully_neutral_pred, _ = getPrediction(fully_neutral_pred_features, model, correctClassIndex, 0, device)

    neutral_features.append(fully_neutral_features)
    predictions.append(fully_neutral_pred)

    max_normalized_pred = 0.0

    for threshold in saliency_thresholds:
        quantile = np.quantile(saliency_map, 1 - threshold)
        patch_mask = saliency_map >= quantile
        patch_mask = np.logical_or(patch_mask, random_mask)
        neutral_features_current = create_neutral_features(features, patch_mask, baseline)

        info = estimate_feature_information(neutral_features_current)
        pred_input = torch.from_numpy(neutral_features_current).unsqueeze(0)
        pred, _ = getPrediction(pred_input, model, correctClassIndex, method, device)

        normalized_info = (info - fully_neutral_info) / (original_features_info - fully_neutral_info)
        normalized_info = np.clip(normalized_info, 0.0, 1.0)
        normalized_pred = (pred - fully_neutral_pred) / (original_pred - fully_neutral_pred)
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        max_normalized_pred = max(max_normalized_pred, normalized_pred)

        if keep_monotonous:
            entropy_pred_tuples.append((normalized_info, max_normalized_pred))
        else:
            entropy_pred_tuples.append((normalized_info, normalized_pred))

        neutral_features.append(neutral_features_current)
        predictions.append(pred)

    entropy_pred_tuples.append((0.0, 0.0))
    entropy_pred_tuples.append((1.0, 1.0))

    info_data, pred_data = zip(*entropy_pred_tuples)
    interp_func = interpolate.interp1d(x=info_data, y=pred_data)

    curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points, endpoint=False)
    curve_y = np.asarray([interp_func(x) for x in curve_x])

    curve_x = np.append(curve_x, 1.0)
    curve_y = np.append(curve_y, 1.0)

    auc = np.trapz(curve_y, curve_x)

    return PicMetricResultBasic(curve_x=curve_x, curve_y=curve_y, auc=auc)