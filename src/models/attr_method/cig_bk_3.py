import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
import torch.nn.functional as F

import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import torch.nn.functional as F
from attr_method._common import PreprocessInputs, call_model_function

class ContrastiveGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")  # torch.Tensor [1, N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features", None)  # torch.Tensor [N, D]
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        x_value = x_value.to(device).float()
        baseline_features = baseline_features.to(device).float()

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        baseline_features = baseline_features.unsqueeze(0)
        attribution_values = torch.zeros_like(x_value, device=device)

        x_diff = x_value - baseline_features
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # Skip alpha=0

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = baseline_features + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True)

            with torch.no_grad():
                logits_r = call_model_function(baseline_features, model, call_model_args)
                if isinstance(logits_r, tuple):
                    logits_r = logits_r[0]

            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            l2_loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss = l2_loss / (l2_loss.item() + 1e-8)

            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=x_step_batch,
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )[0]

            if gradients is None:
                print(f"No gradients at alpha {alpha:.2f}, skipping")
                continue

            counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += counterfactual_gradients

        x_diff_mean = x_diff.mean(dim=0)
        attribution_values *= x_diff_mean
        return attribution_values.detach().cpu().numpy() / x_steps
