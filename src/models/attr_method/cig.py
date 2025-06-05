import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import torch.nn.functional as F
# from attr_method._common import PreprocessInputs, call_model_function

class CIG(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")  # numpy array [1, N, D] or tensor
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features", None)  # numpy array or tensor [N, D]
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert inputs to torch tensors if necessary
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        baseline_features = baseline_features.unsqueeze(0)  # Shape: [1, N, D]
        attribution_values = torch.zeros_like(x_value, device=device)

        x_diff = x_value - baseline_features
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # Skip alpha=0

        # Precompute reference logits

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = (baseline_features + alpha * x_diff).requires_grad_(True)
            x_step_batch.retain_grad()  # <-- add this line 
            with torch.no_grad():
                logits_r = call_model_function(baseline_features, model, call_model_args)
                if isinstance(logits_r, tuple):
                    logits_r = logits_r[0]
 
            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            # Compute L2 loss between step and reference logits
            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()
            print("Leaf:", x_step_batch.is_leaf, "Requires grad:", x_step_batch.requires_grad)
            print("Leaf:", baseline_features.is_leaf, "Requires grad:", baseline_features.requires_grad)
            print("Loss:", loss.item(), "Requires grad:", loss.requires_grad)
            print(">>> x_step_batch.grad", x_step_batch.grad)
            
        #     gradients = torch.autograd.grad(
        #         outputs=loss,
        #         inputs=x_step_batch,
        #         grad_outputs=torch.ones_like(loss),
        #         retain_graph=True,
        #         create_graph=False,
        #         allow_unused=True
        #     )[0]

        #     if gradients is None:
        #         print(f">>>>> No gradients at alpha {alpha:.2f}, skipping")
        #         continue

        #     counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
        #     attribution_values += counterfactual_gradients

        # x_diff_mean = x_diff.mean(dim=0)
        # attribution_values *= x_diff_mean
        # return attribution_values.detach().cpu().numpy() / x_steps
