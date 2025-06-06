import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import torch.nn.functional as F

class CIG(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")  # tensor [1, N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features", None)  # tensor [N, D]
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert inputs to torch tensors if necessary
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        baseline_features = baseline_features.unsqueeze(0)  # Shape: [1, N, D]
        attribution_values = torch.zeros_like(x_value, device=device)
        
        x_diff = x_value - baseline_features
        x_diff_mean = x_diff.mean(dim=0)

        # Define alphas and index 7 intermediate values
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # Skip alpha=0
        alpha_indices = np.linspace(0, x_steps - 1, num=7, dtype=int)
        alpha_plot = alphas[alpha_indices]  # tensor of 7 selected alphas
        visual_attr_list = []

          
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing CIG", ncols=100)):
            x_step_batch = baseline_features + alpha * x_diff  # [1, N, D]
            x_step_batch.requires_grad_(True)
            x_step_batch.retain_grad()

            logits_r = call_model_function(baseline_features, model, call_model_args)
            if isinstance(logits_r, tuple):
                logits_r = logits_r[0]
 
            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]
            

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            gradients = x_step_batch.grad 
            if gradients is None:
                print(f"[WARN] No gradients at alpha {alpha:.3f}, skipping")
                continue
            
            counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += counterfactual_gradients

            if step_idx in alpha_indices:
                intermediate_attr = (counterfactual_gradients * x_diff.squeeze(0)).detach().cpu().numpy().copy()
                visual_attr_list.append(intermediate_attr)

        final_attr = (attribution_values * x_diff_mean).detach().cpu().numpy() / x_steps
        stacked_attrs = np.stack(visual_attr_list) if visual_attr_list else np.zeros((0, *final_attr.shape))

        return {
            "full": final_attr,                    # [N, D]
            "alpha_samples": None, #stacked_attrs,        # [7, N, D]
            "alphas_used": alpha_plot.tolist()     # list of 7 floats
        }
