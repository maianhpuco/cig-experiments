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
        alpha_plot = kwargs.get("alpha_plot", [])
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
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]
        attribution_values = torch.zeros_like(x_value, device=device)

        alpha_visual_attributions = {float(alpha.item()): [] for alpha in alphas if float(alpha.item()) in alpha_plot}

        logits_r = call_model_function(baseline_features, model, call_model_args)
        if isinstance(logits_r, tuple):
            logits_r = logits_r[0]

        for alpha in tqdm(alphas, desc="Computing CIG", ncols=100):
            x_step = baseline_features + alpha * x_diff
            x_step.requires_grad_(True)

            logits_step = call_model_function(x_step, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            gradients = x_step.grad
            if gradients is None:
                continue

            grads = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += grads

            alpha_float = float(alpha.item())
            if alpha_float in alpha_visual_attributions:
                intermediate_attr = grads * x_diff.squeeze(0)
                alpha_visual_attributions[alpha_float].append(intermediate_attr.detach().cpu().numpy())

        x_diff_mean = x_diff.mean(dim=0)
        final_attr = (attribution_values * x_diff_mean).detach().cpu().numpy() / x_steps

        return {
            "full": final_attr,  # [N, D]
            "alpha_samples": {
                alpha: np.stack(v) for alpha, v in alpha_visual_attributions.items()
            }
        }
