import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    # Ensure inputs are a tensor with gradients enabled if needed
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)

    # Run model forward pass
    model.eval()
    with torch.no_grad() if not inputs.requires_grad else torch.enable_grad():
        logits = model(inputs)  # CLAM returns (logits, prob, pred, _, dict)

    # Extract logits from tuple
    logits_tensor = logits[0] if isinstance(logits, tuple) else logits

    # Handle gradient computation for methods like Integrated Gradients
    if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        return {INPUT_OUTPUT_GRADIENTS: gradients}

    # Return class-specific logits for ContrastiveGradients
    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    if logits_tensor.dim() == 2:
        return logits_tensor[:, target_class_idx]
    return logits_tensor


class ContrastiveGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # Expected: torch.Tensor [N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # Expected: torch.Tensor [M, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0) if call_model_args else 0

        # Ensure inputs are on the correct device and type
        x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device, dtype=torch.float32)
        baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device, dtype=torch.float32)

        # Initialize attribution values
        attribution_values = np.zeros_like(x_value.cpu().numpy(), dtype=np.float32)

        # Sample random baseline
        try:
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline_batch = baseline_features[sampled_indices]  # [N, D]
        except (IndexError, ValueError):
            print("Warning: Invalid baseline sampling, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch  # [N, D]

        # Check if x_diff is too small
        if torch.norm(x_diff).item() < 1e-6:
            print("Warning: x_diff is near zero, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)
            x_diff = x_value - x_baseline_batch

        # Ensure model is in evaluation mode
        model.eval()

        alphas = np.linspace(0, 1, x_steps)
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            # Interpolate
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)

            # Baseline forward pass
            x_baseline_torch = x_baseline_batch.clone().detach().requires_grad_(False).to(device)
            logits_x_r = call_model_function(x_baseline_torch, model, call_model_args)

            # Interpolated input forward pass
            logits_x_step = call_model_function(x_step_batch, model, call_model_args)

            # Debug: Check types and shapes
            if not isinstance(logits_x_r, torch.Tensor) or not isinstance(logits_x_step, torch.Tensor):
                print(f"Error: Expected tensors, got logits_x_r: {type(logits_x_r)}, logits_x_step: {type(logits_x_step)}")
                raise TypeError("call_model_function returned non-tensor outputs")
            print(f"Alpha {alpha:.2f}, logits_x_r shape: {logits_x_r.shape}, logits_x_step shape: {logits_x_step.shape}")

            # Compute counterfactual loss: ||logits_diff||²
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2

            # Zero gradients to prevent accumulation
            if x_step_batch.grad is not None:
                x_step_batch.grad.zero_()

            # Backward pass
            logits_difference.backward()

            # Check gradients
            if x_step_batch.grad is None:
                raise RuntimeError("Gradients are not computed! Ensure model parameters require gradients.")

            grad_logits_diff = x_step_batch.grad.cpu().numpy()  # [N, D]
            counterfactual_gradients = grad_logits_diff  # [N, D]
            attribution_values += counterfactual_gradients

            # Debug prints
            print(f"Alpha {alpha:.2f}, Logits diff: {logits_difference.item():.4f}, Grad norm: {torch.norm(x_step_batch.grad):.4f}")

        x_diff = x_diff.cpu().numpy()  # [N, D]
        attribution_values = attribution_values * x_diff  # [N, D]

        result = attribution_values / x_steps

        # Check for zero attributions
        if np.all(result == 0):
            print("Warning: All attribution values are zero. Check model output, baseline, or call_model_function.")

        return result