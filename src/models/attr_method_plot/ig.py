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
        alpha_plot = kwargs.get("alpha_plot", [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
        device = kwargs.get("device", "cpu")

        alphas = np.linspace(0, 1, x_steps)
        alpha_indices = sorted(list(set([
            int(x_steps * a) for a in alpha_plot if 0 < a < 1
        ])))

        all_attributions = []
        visual_attr_list = []

        for s in range(num_samples):
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline = baseline_features[sampled_indices].to(device)
            x_value = x_value.to(device)
            x_diff = x_value - x_baseline

            attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

            for step_idx, alpha in enumerate(tqdm(alphas, desc=f"IG sample {s+1}/{num_samples}", leave=False, ncols=100)):
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

                if step_idx in alpha_indices:
                    intermediate_attr = gradients * x_diff
                    visual_attr_list.append(intermediate_attr.detach().cpu().numpy())

            averaged_attr = (attribution_values * x_diff).detach().cpu().numpy() / x_steps
            all_attributions.append(averaged_attr)

        stacked_full = np.mean(np.stack(all_attributions), axis=0)  # [N, D]
        stacked_visual = np.stack(visual_attr_list) if visual_attr_list else np.zeros((0, *stacked_full.shape))

        return {
            "full": stacked_full,            # shape: [N, D]
            "alpha_samples": stacked_visual  # shape: [7, N, D]
        }
