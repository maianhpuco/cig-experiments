import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs, call_model_function 

 
class ContrastiveGradients(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 25)

        attribution_values = np.zeros_like(x_value, dtype=np.float32)

        alphas = np.linspace(0, 1, x_steps)
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]
        x_diff = x_value - x_baseline_batch

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff

            x_baseline_torch = x_baseline_batch.clone().detach().float()
            x_step_batch_torch = x_step_batch.clone().detach().float().requires_grad_()

            
            logits_r = model(x_baseline_torch, [x_baseline_torch.shape[0]]) 
            logits_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])

            logits_r = logits_r[0] if isinstance(logits_r, tuple) else logits_r
            logits_step = logits_step[0] if isinstance(logits_step, tuple) else logits_step

            target_class_idx = call_model_args.get("target_class_idx", 0)
            diff = torch.norm(logits_step - logits_r, p=2) ** 2
            diff.backward()

            grad = x_step_batch_torch.grad.detach().cpu().numpy()
            counterfactual_gradients = grad.mean(axis=0)

            attribution_values += counterfactual_gradients

        x_diff = x_diff.mean(axis=0)
        attribution_values = attribution_values * x_diff

        return attribution_values / x_steps
