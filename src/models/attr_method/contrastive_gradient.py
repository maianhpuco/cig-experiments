import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs


import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS


class ContrastiveGradients(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]  # expected shape [N, D]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]  # shape [M, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors and move to device
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        attribution_values = torch.zeros_like(x_value, dtype=torch.float32).to(device)
        alphas = torch.linspace(0, 1, x_steps).to(device)

        sampled_indices = torch.randint(0, baseline_features.shape[0], (1, x_value.shape[0]), device=device)
        x_baseline_batch = baseline_features[sampled_indices.squeeze(0)]  # [N, D]
        x_diff = x_value - x_baseline_batch

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff

            x_baseline_torch = x_baseline_batch.clone().detach().to(device)
            x_step_batch_torch = x_step_batch.clone().detach().to(device).requires_grad_(True)

            logits_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])[0]
            print(logits_r)
            logits_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])[0]
            print(logits_step)
            logits_r = logits_r[0] if isinstance(logits_r, tuple) else logits_r
            logits_step = logits_step[0] if isinstance(logits_step, tuple) else logits_step

            target_class_idx = call_model_args.get("target_class_idx", 0)
            if logits_r.dim() == 2:
                logits_r = logits_r[:, target_class_idx]
                logits_step = logits_step[:, target_class_idx]
            else:
                logits_r = logits_r[target_class_idx]
                logits_step = logits_step[target_class_idx]

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            grad = x_step_batch_torch.grad
            attribution_values += grad.mean(dim=0)

        x_diff_mean = x_diff.mean(dim=0)
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        return final_attribution.detach().cpu().numpy()
