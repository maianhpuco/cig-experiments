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

        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]  # [1, N, D]
        x_baseline_batch = x_baseline_batch.squeeze(0)         # [N, D]
        x_diff = x_value - x_baseline_batch                    # [N, D]

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            # ------------ Counterfactual Gradient ------------ 
            x_step_batch = x_baseline_batch + alpha * x_diff   # [N, D]
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)

            logits_x_r = model(x_baseline_batch, [x_baseline_batch.shape[0]])
            logits_x_step = model(x_step_batch, [x_step_batch.shape[0]])

            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            logits_difference.backward()

            if x_step_batch.grad is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff = x_step_batch.grad.detach()  # [N, D]
            counterfactual_gradients = grad_logits_diff.mean(dim=0)  # [D]
            attribution_values += counterfactual_gradients

        x_diff_mean = x_diff.mean(dim=0)  # [D]
        attribution_values = attribution_values * x_diff_mean

        return attribution_values.detach().cpu().numpy() / x_steps



