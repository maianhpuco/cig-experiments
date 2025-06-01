import os
import numpy as np
import torch
from tqdm import tqdm
import saliency.core as saliency
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from attr_method._common import PreprocessInputs


def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)

    logits = model(inputs)  # CLAM returns (logits, prob, pred, _, dict)

    if INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        logits_tensor = logits[0] if isinstance(logits, tuple) else logits
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        return {INPUT_OUTPUT_GRADIENTS: gradients}

    return logits


class IDG(CoreSaliency):
    """Integrated Decision Gradients with slope-weighted redistribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    @staticmethod
    def getSlopes(x_baseline_batch, x_value, model, x_steps, device, target_class_idx=0):
        alphas = torch.linspace(0, 1, x_steps, device=device)
        logits = torch.zeros(x_steps, device=device)
        slopes = torch.zeros(x_steps, device=device)
        x_diff = x_value - x_baseline_batch

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing Slopes", ncols=100)):
            with torch.no_grad():
                x_step_batch = (x_baseline_batch + alpha * x_diff).to(device)
                model_output = model(x_step_batch)
                logits_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
                logit = logits_tensor[0, target_class_idx] if logits_tensor.dim() == 2 else logits_tensor[target_class_idx]
                logits[step_idx] = logit

        delta_alpha = float(alphas[1] - alphas[0])
        slopes[1:] = (logits[1:] - logits[:-1]) / (delta_alpha + 1e-9)
        slopes[0] = 0.0
        return slopes, delta_alpha, logits

    @staticmethod
    def getAlphaParameters(slopes, steps, step_size):
        normed = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + 1e-8)
        normed[0] = 0
        weights = normed / (torch.sum(normed) + 1e-8)

        placements_float = weights * steps
        placements_int = placements_float.floor().int()
        remaining = steps - torch.sum(placements_int)

        float_copy = placements_float.clone()
        float_copy[torch.where(placements_int != 0)] = -1
        top_idx = torch.flip(torch.sort(float_copy)[1], dims=[0])
        placements_int[top_idx[:remaining]] += 1

        alphas = torch.zeros(steps, device=slopes.device)
        alpha_steps = torch.zeros(steps, device=slopes.device)
        start_idx = 0
        start_val = 0

        for count in placements_int:
            if count == 0:
                continue
            count = int(count.item())
            new_alphas = torch.linspace(start_val, start_val + step_size, count + 1, device=slopes.device)[:count]
            alphas[start_idx:start_idx + count] = new_alphas
            alpha_steps[start_idx:start_idx + count] = (step_size / count)
            start_idx += count
            start_val += step_size

        return alphas, alpha_steps

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # (N, D)
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # (M, D)
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0) if call_model_args else 0

        # Device and format checks
        x_value = x_value.to(device, dtype=torch.float32)
        baseline_features = baseline_features.to(device, dtype=torch.float32)

        # Sample baseline with shape (N, D)
        sample_idx = torch.randint(0, baseline_features.size(0), (x_value.size(0),), device=device)
        x_baseline_batch = baseline_features[sample_idx]
        x_diff = x_value - x_baseline_batch

        slopes, _, _ = self.getSlopes(x_baseline_batch, x_value, model, x_steps, device, target_class_idx)
        alphas, alpha_sizes = self.getAlphaParameters(slopes, x_steps, 1.0 / x_steps)

        integrated = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        prev_logit = None
        slope_cache = torch.zeros(x_steps, device=device)

        for step_idx, (alpha, step_size) in enumerate(tqdm(zip(alphas, alpha_sizes), total=x_steps, desc="Computing IG²", ncols=100)):
            x_step = (x_baseline_batch + alpha * x_diff).detach().requires_grad_(True)

            call_output = call_model_function(
                x_step, model, call_model_args=call_model_args, expected_keys=self.expected_keys
            )

            model_output = model(x_step)
            logits_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
            logit = logits_tensor[0, target_class_idx] if logits_tensor.dim() == 2 else logits_tensor[target_class_idx]

            grads = call_output[INPUT_OUTPUT_GRADIENTS]
            grads_avg = torch.tensor(grads, dtype=torch.float32, device=device)

            if prev_logit is not None:
                alpha_diff = alpha - alphas[step_idx - 1]
                slope_cache[step_idx] = (logit - prev_logit) / (alpha_diff + 1e-9)
            prev_logit = logit

            integrated += grads_avg * slope_cache[step_idx] * step_size

            del x_step, grads_avg, call_output, model_output, logits_tensor
            torch.cuda.empty_cache()

        attribution = integrated * x_diff
        return attribution.detach().cpu().numpy()
