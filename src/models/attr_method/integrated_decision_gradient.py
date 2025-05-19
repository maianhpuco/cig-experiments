import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs, call_model_function 


class IntegratedDecisionGradients(CoreSaliency):
    """Integrated Decision Gradients with alpha redistribution and slope-weighted integration."""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    @staticmethod
    def getSlopes(x_baseline_batch, x_value, model, x_steps, device):
        alphas = np.linspace(0, 1, x_steps)
        logits = torch.zeros(x_steps, device=device)
        slopes = torch.zeros(x_steps, device=device)

        x_diff = x_value - x_baseline_batch  # shape: [N, D]

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing IG", ncols=100)):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = x_step_batch.clone().detach().requires_grad_(True).to(device)
            logit = model(x_step_batch_tensor, [x_step_batch_tensor.shape[0]])
            logits[step_idx] = logit.squeeze()

        x_diff_value = float(alphas[1] - alphas[0])
        slopes[1:] = (logits[1:] - logits[:-1]) / x_diff_value
        slopes[0] = 0
        return slopes, x_diff_value, logits

    @staticmethod
    def getAlphaParameters(slopes, steps, step_size):
        slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + 1e-8)
        slopes_0_1_norm[0] = 0
        slopes_sum_1_norm = slopes_0_1_norm / torch.sum(slopes_0_1_norm + 1e-8)

        sample_placements_float = torch.mul(slopes_sum_1_norm, steps)
        sample_placements_int = sample_placements_float.floor().int()
        remaining_to_fill = steps - torch.sum(sample_placements_int)

        non_zeros = torch.where(sample_placements_int != 0)[0]
        sample_placements_float[non_zeros] = -1
        remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims=[0])
        sample_placements_int[remaining_hi_lo[:remaining_to_fill]] += 1

        alphas = torch.zeros(steps)
        alpha_substep_size = torch.zeros(steps)
        alpha_start_index = 0
        alpha_start_value = 0

        for num_samples in sample_placements_int:
            if num_samples == 0:
                continue
            num_samples = int(num_samples.item())
            alphas[alpha_start_index: alpha_start_index + num_samples] = torch.linspace(
                alpha_start_value,
                alpha_start_value + step_size,
                num_samples + 1
            )[:num_samples]
            alpha_substep_size[alpha_start_index: alpha_start_index + num_samples] = (step_size / num_samples)
            alpha_start_index += num_samples
            alpha_start_value += step_size

        return alphas, alpha_substep_size

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value").to(kwargs.get("device", "cpu"))
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features").to(x_value.device)
        x_steps = kwargs.get("x_steps", 25)
        device = x_value.device

        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices].squeeze(0)  # [N, D]
        x_value_flat = x_value.reshape(-1, x_value.shape[-1])
        x_baseline_batch_flat = x_baseline_batch.reshape(-1, x_baseline_batch.shape[-1])

        slopes, x_diff_value, logits = self.getSlopes(x_baseline_batch_flat, x_value_flat, model, x_steps, device)
        new_alphas, alpha_substep_size = self.getAlphaParameters(slopes, x_steps, 1.0 / x_steps)
        alphas_np = new_alphas.detach().cpu().numpy()
        alpha_substep_size_np = alpha_substep_size.detach().cpu().numpy()

        _integrated_gradient = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        prev_logit = None
        slopes_np = np.zeros(x_steps)
        x_diff = x_value - x_baseline_batch  # [N, D]

        for step_idx, alpha in enumerate(tqdm(alphas_np, desc="Computing IG²", ncols=100), start=1):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = x_step_batch.clone().detach().requires_grad_(True).to(device)

            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            logit = model(x_step_batch_tensor, [x_step_batch_tensor.shape[0]]).detach().squeeze()
            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS]  # expected: np.ndarray
            gradients_avg = torch.tensor(gradients_batch, device=device).mean(dim=0)

            idx = step_idx - 1
            if prev_logit is not None:
                alpha_diff = alpha - alphas_np[idx - 1]
                slopes_np[idx] = (logit.item() - prev_logit.item()) / (alpha_diff + 1e-9)
            prev_logit = logit

            weighted_grad = gradients_avg * slopes_np[idx] * alpha_substep_size_np[idx]
            _integrated_gradient += weighted_grad

        attribution_values = _integrated_gradient * x_diff  # element-wise mult
        return attribution_values.detach().cpu().numpy()
