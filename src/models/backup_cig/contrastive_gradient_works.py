import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    """
    Generic model call with gradient support for class attribution.
    """
    device = next(model.parameters()).device
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True).to(device)

    model.eval()
    outputs = model(inputs)  # CLAM returns (logits, prob, pred, _, dict)

    logits_tensor = outputs[0] if isinstance(outputs, tuple) else outputs

    print(f"call_model_function: expected_keys={expected_keys}, inputs shape={inputs.shape}, requires_grad={inputs.requires_grad}, logits_tensor shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}, values={logits_tensor.detach().cpu().numpy()}")

    if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(
            outputs=target_output,
            inputs=inputs,
            grad_outputs=torch.ones_like(target_output),
            create_graph=False,
            retain_graph=False
        )[0]
        print(f"call_model_function: Returning gradients, shape={gradients.shape}")
        return {INPUT_OUTPUT_GRADIENTS: gradients.detach().cpu().numpy()}

    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    if logits_tensor.dim() == 2:
        class_logits = logits_tensor[:, target_class_idx]
        print(f"call_model_function: Returning class {target_class_idx} logits, shape={class_logits.shape}, requires_grad={class_logits.requires_grad}, values={class_logits.detach().cpu().numpy()}")
        return class_logits
    print(f"call_model_function: Returning full logits, shape={logits_tensor.shape}")
    return logits_tensor

class ContrastiveGradients(CoreSaliency):
    """
    Contrastive Gradients Attribution for per-class computation using autograd.
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]  # Expected: torch.Tensor [N, D]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]  # Expected: torch.Tensor [M, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Ensure model parameters require gradients
        for param in model.parameters():
            if not param.requires_grad:
                print("Warning: Some model parameters are frozen, setting requires_grad=True")
                param.requires_grad_(True)

        # Prepare tensors
        x_value = (torch.tensor(x_value, dtype=torch.float32, device=device)
                   if not isinstance(x_value, torch.Tensor) else x_value.to(device, dtype=torch.float32))
        baseline_features = (torch.tensor(baseline_features, dtype=torch.float32, device=device)
                            if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device, dtype=torch.float32))

        # Initialize attribution values
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        # Sample baseline
        try:
            sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
            x_baseline_batch = baseline_features[sampled_indices]  # [N, D]
        except (IndexError, ValueError):
            print("Warning: Invalid baseline sampling, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch  # [N, D]

        # Check x_diff norm
        x_diff_norm = torch.norm(x_diff).item()
        print(f"x_diff norm: {x_diff_norm:.4f}")
        if x_diff_norm < 1e-6:
            print("Warning: x_diff is near zero, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)
            x_diff = x_value - x_baseline_batch
            x_diff_norm = torch.norm(x_diff).item()
            print(f"New x_diff norm: {x_diff_norm:.4f}")

        model.eval()
        alphas = torch.linspace(0, 1, x_steps, device=device)

        for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)
            x_baseline_torch = x_baseline_batch.clone().detach().to(device)

            # Compute logits for baseline (no gradients needed)
            with torch.no_grad():
                logits_r = call_model_function(x_baseline_torch, model, call_model_args)

            # Compute gradients for perturbed input
            call_model_output = call_model_function(
                x_step_batch,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            if not isinstance(logits_r, torch.Tensor):
                print(f"Error: logits_r is not a tensor, got {type(logits_r)}")
                raise TypeError("logits_r must be a tensor")

            # Get gradients
            gradients = call_model_output[INPUT_OUTPUT_GRADIENTS]  # [N, D], numpy
            gradients_torch = torch.tensor(gradients, device=device, dtype=torch.float32)

            # Compute counterfactual contribution
            logits_step = call_model_function(x_step_batch, model, call_model_args)  # [1]
            logits_diff = (logits_step - logits_r).norm().item()
            print(f"Class {target_class_idx}, Alpha {alpha:.2f}, logits_r shape: {logits_r.shape}, logits_step shape: {logits_step.shape}, logits_diff: {logits_diff:.4f}, logits_r: {logits_r.detach().cpu().numpy()}, logits_step: {logits_step.detach().cpu().numpy()}, gradients norm: {torch.norm(gradients_torch):.4f}")

            attribution_values += gradients_torch

        x_diff_mean = x_diff.mean(dim=0)
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        result = final_attribution.detach().cpu().numpy()
        if np.all(result == 0):
            print(f"Warning: All attribution values for class {target_class_idx} are zero. Check model output or baseline.")

        return result