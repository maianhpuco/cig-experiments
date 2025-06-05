import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class CIG(CoreSaliency):
    """
    Contrastive Integrated Gradients Attribution for per-class computation.
    """

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]  # Expected: torch.Tensor [1, N, D]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]  # Expected: torch.Tensor [N, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)
        call_model_function = kwargs["call_model_function"]

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad_(True)

        # Prepare input tensors
        x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device, dtype=torch.float32)
        baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device, dtype=torch.float32)

        # Check dimensions
        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        # Initialize attribution tensor
        attribution_values = torch.zeros_like(x_value, device=device)

        # Expand baseline to match input shape
        x_baseline_batch = baseline_features.unsqueeze(0)  # Shape: [1, N, D]
        x_diff = x_value - x_baseline_batch  # Shape: [1, N, D]

        if torch.norm(x_diff).item() < 1e-6:
            print("x_diff near zero, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value)
            x_diff = x_value - x_baseline_batch

        model.eval()
        alphas = torch.linspace(0, 1, x_steps, device=device)[1:]  # skip alpha=0

        for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
            x_step_batch = (x_baseline_batch + alpha * x_diff).clone().detach().requires_grad_(True)

            # Get baseline logits without tracking gradients
            with torch.no_grad():
                logits_r = call_model_function(x_baseline_batch, model, call_model_args)
                if isinstance(logits_r, tuple):
                    logits_r = logits_r[0]

            # Forward pass with gradient tracking
            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            if not logits_step.requires_grad:
                raise RuntimeError("logits_step does not require gradients")

            # Compute contrastive loss
            loss = torch.norm(logits_step - logits_r, p=2) ** 2

            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=x_step_batch,
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )[0]

            if gradients is not None:
                attribution_values += gradients
            else:
                print(f"No gradients at alpha {alpha:.2f}, skipping")

        x_diff_mean = x_diff.mean(dim=1, keepdim=True)  # Shape: [1, 1, D]
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        result = final_attribution.detach().cpu().numpy()

        if np.all(result == 0):
            print(f"Warning: All attribution values for class {target_class_idx} are zero.")

        return result
