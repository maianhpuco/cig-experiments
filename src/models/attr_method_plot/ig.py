import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
 
class IG(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # (N, D)
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)  # (M, D)
        x_steps = kwargs.get("x_steps", 50)
        num_samples = kwargs.get("num_samples", 5)
        alpha_plot = kwargs.get("alpha_plot", [])  # List of alphas for visualization
        device = kwargs.get("device", "cpu")

        alphas = np.linspace(0, 1, x_steps)
        all_attributions = []
        alpha_visual_attributions = {alpha: [] for alpha in alpha_plot}

        for s in range(num_samples):
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline = baseline_features[sampled_indices].to(device)
            x_value = x_value.to(device)
            x_diff = x_value - x_baseline

            attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

            for alpha in tqdm(alphas, desc=f"Computing sample {s+1}/{num_samples}", leave=False, ncols=100):
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
                attribution_values += gradients

                if alpha in alpha_plot:
                    intermediate_attr = gradients * x_diff
                    alpha_visual_attributions[alpha].append(intermediate_attr.detach().cpu().numpy())

            attr = attribution_values * x_diff
            all_attributions.append(attr.cpu().numpy() / x_steps)

        return {
            "full": np.stack(all_attributions),  # shape: (num_samples, N, D)
            "alpha_samples": {
                alpha: np.stack(v) for alpha, v in alpha_visual_attributions.items()
            }
        }

