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
        baseline_features = kwargs.get("baseline_features")  # (M, D)
        num_samples = kwargs.get("num_samples", 5)
        num_alphas = kwargs.get("x_steps", 50)
        device = kwargs.get("device", "cpu")

        # Get full alphas and 7 sample indices
        alphas = torch.linspace(0, 1, steps=num_alphas, device=device)[1:]  # exclude 0
        sample_indices = np.linspace(0, num_alphas - 2, 7, dtype=int)  # pick 7 evenly
        alpha_plot = alphas[sample_indices]  # [7] tensor

        all_attributions = []
        all_intermediate_attrs = []

        for s in range(num_samples):
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline = baseline_features[sampled_indices].to(device)
            x_value = x_value.to(device)
            x_diff = x_value - x_baseline

            attr_sum = torch.zeros_like(x_value, dtype=torch.float32, device=device)
            intermediate_attrs = []

            for i, alpha in enumerate(tqdm(alphas, desc=f"IG sample {s+1}/{num_samples}", leave=False, ncols=100)):
                x_step = x_baseline + alpha * x_diff
                x_step.requires_grad_(True)

                call_model_output = call_model_function(
                    x_step, model,
                    call_model_args=call_model_args,
                    expected_keys=self.expected_keys
                )

                self.format_and_check_call_model_output(call_model_output, x_step.shape, self.expected_keys)
                gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device)
                attr_sum += gradients

                if i in sample_indices:
                    intermediate_attr = gradients * x_diff
                    intermediate_attrs.append(intermediate_attr.detach().cpu().numpy())

            averaged_attr = (attr_sum * x_diff).detach().cpu().numpy() / len(alphas)
            all_attributions.append(averaged_attr)
            all_intermediate_attrs.append(np.stack(intermediate_attrs))  # [7, N, D]

        stacked_full = np.mean(np.stack(all_attributions), axis=0)       # [N, D]
        stacked_visual = np.mean(np.stack(all_intermediate_attrs), axis=0)  # [7, N, D]

        return {
            "full": stacked_full,
            "alpha_samples": stacked_visual,
            "alphas_used": alpha_plot.tolist()  # return float list
        }
