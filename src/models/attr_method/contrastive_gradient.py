import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class ContrastiveGradients(CoreSaliency):
    """
    Contrastive Gradients Attribution for per-class computation using autograd.
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]
        call_model_function = kwargs.get("call_model_function") or call_model_function 
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Convert input types and move to device
        x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device)
        baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device)

        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        # Sample random baseline
        try:
            sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
            x_baseline_batch = baseline_features[sampled_indices]
        except Exception:
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch
        if torch.norm(x_diff) < 1e-6:
            x_baseline_batch = torch.zeros_like(x_value, device=device)
            x_diff = x_value - x_baseline_batch

        model.eval()
        alphas = torch.linspace(0, 1, x_steps, device=device)

        for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
            x_step_batch = (x_baseline_batch + alpha * x_diff).clone().detach().requires_grad_(True).to(device)

            with torch.no_grad():
                logits_r = call_model_function(x_baseline_batch, model, call_model_args)

            call_model_output = call_model_function(
                x_step_batch,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device, dtype=torch.float32)
            attribution_values += gradients

        x_diff_mean = x_diff.mean(dim=0)
        final_attribution = (attribution_values * x_diff_mean) / x_steps
        
        return final_attribution.detach().cpu().numpy()
