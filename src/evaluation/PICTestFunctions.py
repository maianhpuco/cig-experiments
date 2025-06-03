import numpy as np
import torch
import os
import glob
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
from scipy import interpolate

'''
- create_neutral_features: Uses a binary mask to keep certain patches unchanged and replaces others with a provided baseline.
- generate_random_mask: Creates a 1D mask for patches.
- estimate_feature_information: Uses the mean L2 norm of feature vectors.
- Input Handling: Takes feature tensors [N, D] and saliency scores [N,].
- Baseline: Uses a provided baseline tensor of shape [N, D].
'''

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

# def estimate_feature_information(features: np.ndarray) -> float:
#     return np.linalg.norm(features, ord=2, axis=1).mean()
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
        # fallback: legacy L2 norm
        # return np.linalg.norm(features, ord=2, axis=1).mean()
    
    # Normalize features and reference for cosine similarity
    norm_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    norm_ref = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)
    
    cosine_sim = np.sum(norm_features * norm_ref, axis=1)  # dot product along dim=1
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

# def compute_pic_metric(features: np.ndarray, saliency_map: np.ndarray, random_mask: np.ndarray, 
#                       saliency_thresholds: Sequence[float], method: int, model, device: str,
#                       baseline: np.ndarray, min_pred_value: float = 0.3, 
#                       keep_monotonous: bool = False, num_data_points: int = 1000) -> PicMetricResultBasic:
#     """
#     Computes Performance Information Curve (SIC or AIC) for a single WSI feature set.
    
#     Args:
#         features: Feature tensor of shape [N, D] (e.g., D=1024).
#         saliency_map: Saliency scores of shape [N,].
#         random_mask: Binary mask of shape [N,].
#         saliency_thresholds: Fractions of important patches to reveal.
#         method: 0 for SIC, 1 for AIC.
#         model: CLAM model (wrapped in ModelWrapper internally).
#         device: Device for computation.
#         baseline: Baseline feature tensor of shape [N, D].
#         min_pred_value: Minimum prediction confidence for original features.
#         keep_monotonous: Whether to enforce monotonicity in the curve.
#         num_data_points: Number of data points for the interpolated curve.
    
#     Returns:
#         PicMetricResultBasic containing the curve and AUC.
#     """
#     model_wrapper = ModelWrapper(model, model_type='clam')
#     neutral_features = []
#     predictions = []
#     entropy_pred_tuples = []

#     # Validate baseline shape
#     if baseline.shape != features.shape:
#         raise ValueError(f"Baseline shape {baseline.shape} must match features shape {features.shape}")
#     print(f"> PIC baseline shape: {baseline.shape}")
#     print(f"> PIC baseline stats: mean={baseline.mean():.6f}, std={baseline.std():.6f}")

#     # Estimate information content
#     original_features_info = estimate_feature_information(features, reference=features)
#     fully_neutral_features = create_neutral_features(features, random_mask, baseline)
#     fully_neutral_info = estimate_feature_information(fully_neutral_features, reference=features)

#     # Original prediction
#     input_features = torch.from_numpy(features).unsqueeze(0)
#     original_pred, correctClassIndex = getPrediction(input_features, model_wrapper, -1, method, device)
#     print(f"{'SIC' if method == 0 else 'AIC'} - Original prediction: {original_pred:.6f} (Class: {correctClassIndex})")

#     if original_pred < min_pred_value:
#         raise ComputePicMetricError(f"Original prediction {original_pred} is below min_pred_value {min_pred_value}")

#     # Fully neutral prediction
#     fully_neutral_pred_features = torch.from_numpy(fully_neutral_features).unsqueeze(0)
#     fully_neutral_pred, _ = getPrediction(fully_neutral_pred_features, model_wrapper, correctClassIndex, method, device)
#     print(f"{'SIC' if method == 0 else 'AIC'} - Fully neutral prediction: {fully_neutral_pred:.6f}")

#     neutral_features.append(fully_neutral_features)
#     predictions.append(fully_neutral_pred)

#     max_normalized_pred = 0.0

#     for threshold in saliency_thresholds:
#         quantile = np.quantile(saliency_map, 1 - threshold)
#         patch_mask = saliency_map >= quantile
#         ''' randomly masked patches (used for smoothing or as a reference) are also included in the patch mask. '''
#         patch_mask = np.logical_or(patch_mask, random_mask)
#         visible_fraction = patch_mask.sum() / len(patch_mask) 
#         print(f">> Threshold {threshold:.3f} — {visible_fraction * 100:.2f}% patches visible")

#         neutral_features_current = create_neutral_features(features, patch_mask, baseline)
        
#         info = estimate_feature_information(neutral_features_current, reference=features)
#         # info = estimate_feature_information(neutral_features_current)
#         print("Information content of neutral features:", info) 
#         pred_input = torch.from_numpy(neutral_features_current).unsqueeze(0)
#         pred, _ = getPrediction(pred_input, model_wrapper, correctClassIndex, method, device)
        
#         print(f"{'SIC' if method == 0 else 'AIC'} - Threshold {threshold:.3f}: Prediction = {pred:.6f}. Info = {info:.4f}")

#         normalized_info = (info - fully_neutral_info) / (original_features_info - fully_neutral_info)
#         print(f"Normalized info: {normalized_info:.4f} (fully neutral: {fully_neutral_info:.4f}, original: {original_features_info:.4f})")
#         normalized_info = np.clip(normalized_info, 0.0, 1.0)
#         normalized_pred = (pred - fully_neutral_pred) / (original_pred - fully_neutral_pred)
#         normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
#         max_normalized_pred = max(max_normalized_pred, normalized_pred)
#         print(f"Normalized prediction: {normalized_pred:.4f} (fully neutral: {fully_neutral_pred:.4f}, original: {original_pred:.4f})")
#         if keep_monotonous:
#             entropy_pred_tuples.append((normalized_info, max_normalized_pred))
#         else:
#             entropy_pred_tuples.append((normalized_info, normalized_pred))

#         neutral_features.append(neutral_features_current)
#         predictions.append(pred)
#     print(f">> Max normalized prediction: {max_normalized_pred:.4f}")
#     entropy_pred_tuples.append((0.0, 0.0))
#     entropy_pred_tuples.append((1.0, 1.0))

#     info_data, pred_data = zip(*entropy_pred_tuples)
#     print(f"Curve points: {list(zip(info_data, pred_data))}")
#     interp_func = interpolate.interp1d(x=info_data, y=pred_data)

#     curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points, endpoint=False)
#     curve_y = np.asarray([interp_func(x) for x in curve_x])

#     curve_x = np.append(curve_x, 1.0)
#     curve_y = np.append(curve_y, 1.0)

#     auc = np.trapz(curve_y, curve_x)

#     return PicMetricResultBasic(curve_x=curve_x, curve_y=curve_y, auc=auc)

def compute_pic_metric(features: np.ndarray, saliency_map: np.ndarray, random_mask: np.ndarray, 
                      saliency_thresholds: Sequence[float], method: int, model, device: str,
                      baseline: np.ndarray, min_pred_value: float = 0.3, 
                      keep_monotonous: bool = False, num_data_points: int = 1000) -> PicMetricResultBasic:
    """
    Computes Performance Information Curve (SIC or AIC) for a single WSI feature set.
    
    Args:
        features: Feature tensor of shape [N, D] (e.g., D=1024).
        saliency_map: Saliency scores of shape [N,].
        random_mask: Binary mask of shape [N,].
        saliency_thresholds: Fractions of important patches to reveal.
        method: 0 for SIC, 1 for AIC.
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

    # Validate baseline shape
    if baseline.shape != features.shape:
        raise ValueError(f"Baseline shape {baseline.shape} must match features shape {features.shape}")
    # print(f"> PIC baseline shape: {baseline.shape}")
    # print(f"> PIC baseline stats: mean={baseline.mean():.6f}, std={baseline.std():.6f}")

    # Debug saliency map
    # print(f"> Saliency map stats: min={saliency_map.min():.6f}, max={saliency_map.max():.6f}, mean={saliency_map.mean():.6f}, std={saliency_map.std():.6f}")
    # np.save("saliency_map.npy", saliency_map)

    # Estimate information content
    original_features_info = estimate_feature_information(features, reference=features)
    fully_neutral_features = create_neutral_features(features, random_mask, baseline)
    fully_neutral_info = estimate_feature_information(fully_neutral_features, reference=features)

    # Original prediction
    input_features = torch.from_numpy(features).unsqueeze(0).to(device)
    original_pred, correctClassIndex = getPrediction(input_features, model_wrapper, -1, method, device)
    # print(f"{'SIC' if method == 0 else 'AIC'} - Original prediction: {original_pred:.6f} (Class: {correctClassIndex})")

    if original_pred < min_pred_value:
        raise ComputePicMetricError(f"Original prediction {original_pred} is below min_pred_value {min_pred_value}")

    # Fully neutral prediction
    fully_neutral_pred_features = torch.from_numpy(fully_neutral_features).unsqueeze(0).to(device)
    fully_neutral_pred, _ = getPrediction(fully_neutral_pred_features, model_wrapper, correctClassIndex, method, device)
    # print(f"{'SIC' if method == 0 else 'AIC'} - Fully neutral prediction: {fully_neutral_pred:.6f}")

    neutral_features.append(fully_neutral_features)
    predictions.append(fully_neutral_pred)

    max_normalized_pred = 0.0

    for threshold in saliency_thresholds:
        quantile = np.quantile(saliency_map, 1 - threshold)
        patch_mask = saliency_map >= quantile
        # Use random_mask only if fraction > 0
        if random_mask.sum() > 0:
            patch_mask = np.logical_or(patch_mask, random_mask)
        visible_fraction = patch_mask.sum() / len(patch_mask)
        print(f">> Threshold {threshold:.3f} — {visible_fraction * 100:.2f}% patches visible")

        neutral_features_current = create_neutral_features(features, patch_mask, baseline)
        
        info = estimate_feature_information(neutral_features_current, reference=features)
        print(f"Information content of neutral features: {info:.4f}")
        pred_input = torch.from_numpy(neutral_features_current).unsqueeze(0).to(device)
        pred, _ = getPrediction(pred_input, model_wrapper, correctClassIndex, method, device)
        
        normalized_info = (info - fully_neutral_info) / (original_features_info - fully_neutral_info)
        normalized_info = np.clip(normalized_info, 0.0, 1.0)
        normalized_pred = (pred - fully_neutral_pred) / (original_pred - fully_neutral_pred) if (original_pred - fully_neutral_pred) > 1e-6 else pred
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        max_normalized_pred = max(max_normalized_pred, normalized_pred)

        print(f"{'SIC' if method == 0 else 'AIC'} - Threshold {threshold:.3f}: Prediction = {pred:.6f}, Normalized Info = {normalized_info:.4f}, Normalized Pred = {normalized_pred:.4f}")

        if keep_monotonous:
            entropy_pred_tuples.append((normalized_info, max_normalized_pred))
        else:
            entropy_pred_tuples.append((normalized_info, normalized_pred))

        neutral_features.append(neutral_features_current)
        predictions.append(pred)

    entropy_pred_tuples.append((0.0, 0.0))
    entropy_pred_tuples.append((1.0, 1.0))

    # Sort tuples for interpolation
    sorted_tuples = sorted(entropy_pred_tuples, key=lambda x: x[0])
    info_data, pred_data = zip(*sorted_tuples)
    print(f"Curve points: {list(zip(info_data, pred_data))}")

    interp_func = interpolate.interp1d(x=info_data, y=pred_data, fill_value=(0.0, 1.0), bounds_error=False)

    curve_x = np.linspace(0.0, 1.0, num_data_points)
    curve_y = np.asarray([interp_func(x) for x in curve_x])

    auc = np.trapz(curve_y, curve_x)

    # # Plot curve
    # import matplotlib.pyplot as plt
    # plt.plot(curve_x, curve_y, label=f"{'SIC' if method == 0 else 'AIC'} AUC={auc:.3f}")
    # plt.scatter(info_data, pred_data, color='red', label='Data points')
    # plt.xlabel("Normalized Information")
    # plt.ylabel("Normalized Prediction")
    # plt.legend()
    # plt.savefig(f"pic_curve_{'sic' if method == 0 else 'aic'}_class_{correctClassIndex}.png")
    # plt.close()

    return PicMetricResultBasic(curve_x=curve_x, curve_y=curve_y, auc=auc)