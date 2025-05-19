import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs 

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor) 
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)
    
    # CLAM model expects a single bag of features (N, D)
    logits = model(inputs)  # Shape: (B, 2) or (2,)
    
    if INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        if logits.dim() == 2:
            target_output = logits[:, target_class_idx].sum()
        else:
            target_output = logits[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        return {INPUT_OUTPUT_GRADIENTS: gradients}
    
    return logits

class IntegratedDecisionGradients(CoreSaliency):
    """Integrated Decision Gradients with alpha redistribution and slope-weighted integration."""
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    @staticmethod
    def getSlopes(x_baseline_batch, x_value, model, x_steps, device, target_class_idx=0):
        alphas = torch.linspace(0, 1, x_steps, device=device)
        logits = torch.zeros(x_steps, device=device)
        slopes = torch.zeros(x_steps, device=device)

        x_diff = x_value - x_baseline_batch  # [N, D]

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing Slopes", ncols=100)):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True)

            model_output = model(x_step_batch)  # Single bag, shape (N, D)
            print("====== debug ======")
            print(len(model_output))
            
            logit = model_output[0, target_class_idx] if model_output.dim() == 2 else model_output[target_class_idx]
            logits[step_idx] = logit
            print("==== DEBUG ==== ")
            print("model_output shape:", model_output.shape)
            print("logit shape:", logit.shape) 
        x_diff_value = float(alphas[1] - alphas[0])
        slopes[1:] = (logits[1:] - logits[:-1]) / (x_diff_value + 1e-9)
        slopes[0] = 0
        
        return slopes, x_diff_value, logits

    @staticmethod
    def getAlphaParameters(slopes, steps, step_size):
        slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + 1e-8)
        slopes_0_1_norm[0] = 0
        slopes_sum_1_norm = slopes_0_1_norm / (torch.sum(slopes_0_1_norm) + 1e-8)

        sample_placements_float = slopes_sum_1_norm * steps
        sample_placements_int = sample_placements_float.floor().int()
        remaining_to_fill = steps - torch.sum(sample_placements_int)

        non_zeros = torch.where(sample_placements_int != 0)[0]
        sample_placements_float[non_zeros] = -1
        remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims=[0])
        sample_placements_int[remaining_hi_lo[:remaining_to_fill]] += 1

        alphas = torch.zeros(steps, device=slopes.device)
        alpha_substep_size = torch.zeros(steps, device=slopes.device)
        alpha_start_index = 0
        alpha_start_value = 0

        for num_samples in sample_placements_int:
            if num_samples == 0:
                continue
            num_samples = int(num_samples.item())
            alphas[alpha_start_index: alpha_start_index + num_samples] = torch.linspace(
                alpha_start_value,
                alpha_start_value + step_size,
                num_samples + 1,
                device=slopes.device
            )[:num_samples]
            alpha_substep_size[alpha_start_index: alpha_start_index + num_samples] = (step_size / num_samples)
            alpha_start_index += num_samples
            alpha_start_value += step_size

        return alphas, alpha_substep_size

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # Shape: (N, D)
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # Shape: (M, D)
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0) if call_model_args else 0

        # Ensure inputs are tensors on the correct device
        x_value = (x_value.to(device, dtype=torch.float32) if isinstance(x_value, torch.Tensor) 
                   else torch.tensor(x_value, dtype=torch.float32, device=device))
        baseline_features = (baseline_features.to(device, dtype=torch.float32) if isinstance(baseline_features, torch.Tensor) 
                             else torch.tensor(baseline_features, dtype=torch.float32, device=device))

        # Sample baseline features with shape (N, D)
        sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
        x_baseline_batch = baseline_features[sampled_indices]  # [N, D]

        slopes, x_diff_value, _ = self.getSlopes(
            x_baseline_batch,
            x_value,
            model,
            x_steps,
            device,
            target_class_idx
        )

        new_alphas, alpha_substep_size = self.getAlphaParameters(slopes, x_steps, 1.0 / x_steps)
        x_diff = x_value - x_baseline_batch  # [N, D]
        integrated_gradient = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        prev_logit = None
        slopes = torch.zeros(x_steps, device=device)

        for step_idx, (alpha, substep_size) in enumerate(tqdm(
            zip(new_alphas, alpha_substep_size), total=x_steps, desc="Computing IG²", ncols=100
        )):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True)
            print("x_step_batch shape:", x_step_batch.shape, "device:", x_step_batch.device)
            call_model_output = call_model_function(
                x_step_batch,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            model_output = model(x_step_batch)
            logit = model_output[0, target_class_idx] if model_output.dim() == 2 else model_output[target_class_idx]
            logit = logit.detach()

            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS]  # Shape: (N, D)
            gradients_avg = gradients_batch  # Already averaged if necessary
            print("gradients_batch shape:", gradients_batch.shape) 
            if prev_logit is not None:
                alpha_diff = alpha - new_alphas[step_idx - 1]
                slopes[step_idx] = (logit - prev_logit) / (alpha_diff + 1e-9)
            prev_logit = logit

            weighted_grad = gradients_avg * slopes[step_idx] * substep_size
            integrated_gradient += weighted_grad

        attribution_values = integrated_gradient * x_diff  # Element-wise
        return attribution_values 