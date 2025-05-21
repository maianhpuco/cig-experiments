import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs

class ContrastiveGradients(CoreSaliency):
    """Contrastive Gradient Attribution using counterfactual loss."""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # torch.Tensor [N, D]
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # torch.Tensor
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0) if call_model_args else 0

        # Ensure inputs are on the correct device
        x_value = x_value.to(device, dtype=torch.float32)
        baseline_features = baseline_features.to(device, dtype=torch.float32)

        # Initialize attribution values
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        alphas = np.linspace(0, 1, x_steps)

        # Sample random baseline or use zero baseline if sampling fails
        try:
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline_batch = baseline_features[sampled_indices]  # [N, D]
        except (IndexError, ValueError):
            print("Warning: Invalid baseline sampling, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch  # [N, D]

        # Check if x_diff is too small
        if torch.norm(x_diff).item() < 1e-6:
            print("Warning: x_diff is near zero, attributions may be zero")
            x_baseline_batch = torch.zeros_like(x_value, device=device)
            x_diff = x_value - x_baseline_batch

        for alpha in tqdm(alphas, desc="Computing Contrastive Gradients", ncols=100):
            # Interpolate
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)

            # Ensure model is in evaluation mode
            model.eval()

            # Forward passes
            logits_r = model(x_baseline_batch)
            logits_step = model(x_step_batch)

            # Handle tuple outputs and extract class-specific logits
            logits_r = logits_r[0] if isinstance(logits_r, tuple) else logits_r
            logits_step = logits_step[0] if isinstance(logits_step, tuple) else logits_step

            # Select target class logits
            if logits_r.dim() > 1:
                logits_r = logits_r[:, target_class_idx]
                logits_step = logits_step[:, target_class_idx]

            # Compute contrastive loss: ||logits_diff||²
            logits_difference = torch.norm(logits_step - logits_r, p=2) ** 2

            # Zero gradients to prevent accumulation
            if x_step_batch.grad is not None:
                x_step_batch.grad.zero_()

            # Backward pass
            logits_difference.backward()

            # Check gradients
            if x_step_batch.grad is None:
                raise RuntimeError("Gradients are not computed! Ensure model parameters require gradients.")

            grad_logits_diff = x_step_batch.grad.detach()  # [N, D]
            counterfactual_gradients = grad_logits_diff.mean(dim=0)  # [D]
            attribution_values += counterfactual_gradients

            # Debug gradient norm
            # print(f"Alpha {alpha:.2f}, Logits diff: {logits_difference.item():.4f}, Grad norm: {torch.norm(grad_logits_diff):.4f}")

        # Apply attribution
        x_diff_mean = x_diff.mean(dim=0)  # [D]
        attribution_values = attribution_values * x_diff_mean  # [D]

        # Normalize and return as NumPy array
        result = attribution_values.detach().cpu().numpy() / x_steps

        # Check for zero attributions
        if np.all(result == 0):
            print("Warning: All attribution values are zero. Check model output or baseline.")

        return result