import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class CIG(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # [1, N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features")  # [N, D]
        x_steps = kwargs.get("x_steps", 50)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError("Baseline shape does not match input shape.")

        baseline_features = baseline_features.unsqueeze(0)  # [1, N, D]
        x_diff = x_value - baseline_features
        x_diff_mean = x_diff.mean(dim=0)

        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # exclude 0
        alpha_indices = np.linspace(0, x_steps - 1, num=7, dtype=int)
        alpha_plot = alphas[alpha_indices]  # [7]

        attribution_values = torch.zeros_like(x_value, device=device)
        visual_attr_list = []

        # Detach logits_r to avoid retaining graph
        with torch.no_grad():
            logits_r = call_model_function(baseline_features, model, call_model_args)
            if isinstance(logits_r, tuple):
                logits_r = logits_r[0]

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing CIG", ncols=100)):
            x_step = baseline_features + alpha * x_diff
            x_step.requires_grad_(True)

            logits_step = call_model_function(x_step, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            gradients = x_step.grad
            if gradients is None:
                print(f"[WARN] No gradients at step {step_idx}, alpha={alpha.item():.3f}")
                continue

            grads = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += grads

            if step_idx in alpha_indices:
                intermediate_attr = grads * x_diff.squeeze(0)
                visual_attr_list.append(intermediate_attr.detach().cpu().numpy())

        final_attr = (attribution_values * x_diff_mean).detach().cpu().numpy() / x_steps
        stacked_attrs = np.stack(visual_attr_list) if visual_attr_list else np.zeros((0, *final_attr.shape))

        return {
            "full": final_attr,                    # [N, D]
            "alpha_samples": stacked_attrs,        # [7, N, D]
            "alphas_used": alpha_plot.tolist()     # list of 7 floats
        }
