import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs, call_model_function

class ContrastiveGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # Expected: torch.Tensor [N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # Expected: torch.Tensor
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
            sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
            x_baseline_batch = baseline_features[sampled_indices]  # [1, N, D]
        except (IndexError, ValueError):
            print("Warning: Invalid baseline sampling, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device).unsqueeze(0)

        x_diff = x_value - x_baseline_batch  # [1, N, D]

        # Check if x_diff is too small
        if torch.norm(x_diff).item() < 1e-6:
            print("Warning: x_diff is near zero, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device).unsqueeze(0)
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

            # Handle tuple outputs and select target class
            logits_x_r = logits_x_r[0] if isinstance(logits_x_r, tuple) else logits_x_r
            logits_x_step = logits_x_step[0] if isinstance(logits_x_step, tuple) else logits_x_step
            if logits_x_r.dim() > 1:
                logits_x_r = logits_x_r[:, target_class_idx]
                logits_x_step = logits_x_step[:, target_class_idx]

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

            grad_logits_diff = x_step_batch.grad.cpu().numpy()  # [1, N, D]
            counterfactual_gradients = grad_logits_diff.mean(axis=0)  # [N, D]
            attribution_values += counterfactual_gradients

            # Debug prints (uncomment to diagnose)
            print(f"Alpha {alpha:.2f}, Logits diff: {logits_difference.item():.4f}, Grad norm: {torch.norm(x_step_batch.grad):.4f}")

        x_diff = x_diff.mean(axis=0).cpu().numpy()  # [N, D]
        attribution_values = attribution_values * x_diff  # [N, D]

        result = attribution_values / x_steps

        # Check for zero attributions
        if np.all(result == 0):
            print("Warning: All attribution values are zero. Check model output, baseline, or call_model_function.")

        return result