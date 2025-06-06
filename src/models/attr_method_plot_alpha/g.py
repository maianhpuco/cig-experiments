import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class VanillaGradients(CoreSaliency):
    """Vanilla Gradient Attribution with Alpha Sampling and Sparse Visualization"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # [N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # [M, D]
        num_alphas = kwargs.get("num_alphas", 50)
        device = kwargs.get("device", "cpu")

        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        # Sample one baseline per input row
        sampled_idx = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
        x_baseline = baseline_features[sampled_idx]  # [N, D]
        x_diff = x_value - x_baseline

        alphas = torch.linspace(0, 1, steps=num_alphas, device=device)
        alpha_indices = np.linspace(0, num_alphas - 1, num=7, dtype=int)  # [7 indices]
        alpha_plot = alphas[alpha_indices]  # [7] float values

        attribution_sum = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        visual_attr_list = []

        for i, alpha in enumerate(tqdm(alphas, desc="VanillaGrad Alpha Steps", ncols=100)):
            x_interp = x_baseline + alpha * x_diff
            x_interp.requires_grad_(True)

            call_model_output = call_model_function(
                x_interp,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_interp.shape, self.expected_keys)

            gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device)
            attribution_sum += gradients

            if i in alpha_indices:
                intermediate_attr = gradients * x_diff
                visual_attr_list.append(intermediate_attr.detach().cpu().numpy())

        averaged_attr = (attribution_sum * x_diff) / num_alphas  # [N, D]
        stacked_attrs = np.stack(visual_attr_list) if visual_attr_list else np.zeros((0, *averaged_attr.shape))

        return {
            "full": averaged_attr.detach().cpu().numpy(),         # [N, D]
            "alpha_samples": stacked_attrs,                       # [7, N, D]
            "alphas_used": alpha_plot.tolist()                    # [7 floats]
        }
