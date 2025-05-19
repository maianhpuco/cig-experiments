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
    """Contrastive Gradient Attribution using counterfactual loss."""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")                      # torch.Tensor
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # torch.Tensor
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cpu")  

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        alphas = np.linspace(0, 1, x_steps)

        # Sample random baseline for each patch
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices].squeeze(0)  # [N, D]
        x_diff = x_value - x_baseline_batch                                # [N, D]

        for alpha in tqdm(alphas, desc="Computing Contrastive Gradients", ncols=100):
            # Interpolate
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)

            # Forward passes
            logits_r = model(x_baseline_batch)
            logits_step = model(x_step_batch)

            # Safely extract logits from tuple
            logits_r = logits_r[0] if isinstance(logits_r, tuple) else logits_r
            logits_step = logits_step[0] if isinstance(logits_step, tuple) else logits_step

            # Compute contrastive loss: ||logits_diff||²
            logits_difference = torch.norm(logits_step - logits_r, p=2) ** 2
            logits_difference.backward()

            # Get gradients
            if x_step_batch.grad is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff = x_step_batch.grad.detach()  # [N, D]
            counterfactual_gradients = grad_logits_diff.mean(dim=0)  # [D]
            attribution_values += counterfactual_gradients

        # Apply attribution
        x_diff_mean = x_diff.mean(dim=0)  # [D]
        attribution_values = attribution_values * x_diff_mean  # [D]

        return attribution_values.detach().cpu().numpy() / x_steps


