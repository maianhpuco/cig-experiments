import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS


def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    """
    Generic model call with gradient support for class attribution.
    """
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)

    logits = model(inputs)  # CLAM returns (logits, prob, pred, _, dict)

    if INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        logits_tensor = logits[0] if isinstance(logits, tuple) else logits
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        return {INPUT_OUTPUT_GRADIENTS: gradients}

    return logits


class ContrastiveGradients(CoreSaliency):
    """
    Contrastive Gradients Attribution.
    """

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]  # [N, D]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]  # [M, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Prepare tensors
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        attribution_values = torch.zeros_like(x_value, dtype=torch.float32).to(device)
        alphas = torch.linspace(0, 1, x_steps).to(device)

        # Sample baseline indices
        sampled_indices = torch.randint(0, baseline_features.shape[0], (1, x_value.shape[0]), device=device)
        x_baseline_batch = baseline_features[sampled_indices.squeeze(0)]
        x_diff = x_value - x_baseline_batch

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff

            x_baseline_torch = x_baseline_batch.clone().detach().to(device)
            x_step_batch_torch = x_step_batch.clone().detach().to(device).requires_grad_(True)

            # Forward pass
            logits_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])[0]
            logits_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])[0]

            # Get logits per class
            target_class_idx = call_model_args.get("target_class_idx", 0)
            if logits_r.dim() == 2:
                logits_r = logits_r[:, target_class_idx]
                logits_step = logits_step[:, target_class_idx]
            else:
                logits_r = logits_r[target_class_idx]
                logits_step = logits_step[target_class_idx]

            # Compute difference loss and gradients
            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            grad = x_step_batch_torch.grad
            attribution_values += grad.mean(dim=0)

        x_diff_mean = x_diff.mean(dim=0)
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        return final_attribution.detach().cpu().numpy()
