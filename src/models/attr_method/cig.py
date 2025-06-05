import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

import torch
import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

class ContrastiveGradients(CoreSaliency):
    """
    Contrastive Integrated Gradients Attribution for per-class computation.
    """

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")                         # [N, D] or [1, N, D]
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")     # [N, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensor and move to device
        x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device)
        baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device)

        # Ensure input shape is [1, N, D]
        if x_value.dim() == 2:
            x_value = x_value.unsqueeze(0)  # [1, N, D]
        if baseline_features.dim() == 2:
            baseline_features = baseline_features.unsqueeze(0)  # [1, N, D]

        N, D = x_value.shape[1], x_value.shape[2]

        # Sample 1 baseline if not matching shape
        if baseline_features.shape[1] != N or baseline_features.shape[2] != D:
            sampled_indices = torch.randint(0, baseline_features.shape[1], (N,), device=device)
            x_baseline_batch = baseline_features[0, sampled_indices].unsqueeze(0)  # [1, N, D]
        else:
            x_baseline_batch = baseline_features  # [1, N, D]

        x_diff = x_value - x_baseline_batch
        attribution_values = torch.zeros_like(x_value, device=device)

        model.eval()
        alphas = torch.linspace(0, 1, x_steps, device=device)

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            x_step_batch = (x_baseline_batch + alpha * x_diff).detach().requires_grad_(True)  # [1, N, D]

            logits_r = model(x_baseline_batch.squeeze(0), [N])
            logits_step = model(x_step_batch.squeeze(0), [N])

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            if x_step_batch.grad is None:
                print(f"No gradients at alpha {alpha.item():.2f}, skipping")
                continue

            attribution_values += x_step_batch.grad

        x_diff_mean = x_diff.mean(dim=1, keepdim=True)  # [1, 1, D]
        final_attr = (attribution_values * x_diff_mean) / x_steps

        return final_attr.detach().cpu().numpy()
