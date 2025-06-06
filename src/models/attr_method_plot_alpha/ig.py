import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class IG(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # [N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # [M, D]
        x_steps = kwargs.get("x_steps", 50)
        device = kwargs.get("device", "cpu")

        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        # Sample 1 baseline per input row
        sampled_idx = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
        x_baseline = baseline_features[sampled_idx]  # [N, D]
        x_diff = x_value - x_baseline

        alphas = torch.linspace(0, 1, steps=x_steps + 1, device=device)[1:]  # (exclude 0)
        alpha_indices = np.linspace(0, x_steps - 1, num=7, dtype=int)        # 7 indices
        alpha_plot = alphas[alpha_indices]                                   # [7] alphas used

        attribution_sum = torch.zeros_like(x_value, device=device)
        visual_attr_list = []

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing IG", ncols=100)):
            x_step = x_baseline + alpha * x_diff
            x_step.requires_grad_(True)

            call_model_output = call_model_function(
                x_step,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            self.format_and_check_call_model_output(call_model_output, x_step.shape, self.expected_keys)
            gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device)

            attribution_sum += gradients

            if step_idx in alpha_indices:
                intermediate_attr = gradients * x_diff
                visual_attr_list.append(intermediate_attr.detach().cpu().numpy())

        final_attr = (attribution_sum * x_diff).detach().cpu().numpy() / x_steps
        stacked_visuals = np.stack(visual_attr_list) if visual_attr_list else np.zeros((0, *final_attr.shape))

        return {
            "full": final_attr,                      # [N, D]
            "alpha_samples": stacked_visuals,        # [7, N, D]
            "alphas_used": alpha_plot.tolist()       # [7] floats
        }
