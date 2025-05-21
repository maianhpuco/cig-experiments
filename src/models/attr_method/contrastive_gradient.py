import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    """
    Generic model call with gradient support for class attribution.
    """
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)

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
        gradients = torch.autograd.grad(target_output, inputs)[0]
        print("call_model_function: Returning gradients, shape={gradients.shape}")
        return {INPUT_OUTPUT_GRADIENTS: gradients}

    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    if logits_tensor.dim() == 2:
        class_logits = logits_tensor[:, target_class_idx]
        print(f"call_model_function: Returning class {target_class_idx} logits, shape={class_logits.shape}, requires_grad={class_logits.requires_grad}, values={class_logits.detach().cpu().numpy()}")
        return class_logits
    print(f"call_model_function: Returning full logits, shape={logits_tensor.shape}")
    return logits_tensor

# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
# from attr_method._common import PreprocessInputs, call_model_function

class ContrastiveGradients(CoreSaliency):
    """
    Contrastive Gradients Attribution for per-class computation.
    """

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

        # Sample baseline indices
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

        # Gradient hook to debug
        def gradient_hook(module, grad_input, grad_output):
            print(f"Gradient hook: module={module.__class__.__name__}, grad_output[0] shape={grad_output[0].shape if grad_output[0] is not None else None}, grad_output[0] requires_grad={grad_output[0].requires_grad if grad_output[0] is not None else False}")

        # Register hook on model's classifier layer (adjust based on CLAM model structure)
        for name, module in model.named_modules():
            if "classifier" in name.lower() or "fc" in name.lower():
                module.register_backward_hook(gradient_hook)
                print(f"Registered gradient hook on {name}")

        for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)
            x_baseline_torch = x_baseline_batch.clone().detach().to(device)

            # Forward pass
            logits_r = call_model_function(x_baseline_torch, model, call_model_args)
            logits_step = call_model_function(x_step_batch, model, call_model_args)

            # Check types and requires_grad
            if not isinstance(logits_r, torch.Tensor) or not isinstance(logits_step, torch.Tensor):
                print(f"Error: Expected tensors, got logits_r: {type(logits_r)}, logits_step: {type(logits_step)}")
                raise TypeError("call_model_function returned non-tensor outputs")
            if not logits_step.requires_grad:
                print(f"Error: logits_step does not require gradients for class {target_class_idx}, alpha {alpha:.2f}")
                raise RuntimeError("logits_step detached from computation graph")

            logits_diff = (logits_step - logits_r).norm().item()
            print(f"Class {target_class_idx}, Alpha {alpha:.2f}, logits_r shape: {logits_r.shape}, logits_step shape: {logits_step.shape}, logits_diff: {logits_diff:.4f}, logits_r: {logits_r.detach().cpu().numpy()}, logits_step: {logits_step.detach().cpu().numpy()}, logits_step requires_grad: {logits_step.requires_grad}")

            # Compute counterfactual loss
            loss = torch.norm(logits_step - logits_r, p=2) ** 2

            # Zero gradients
            if x_step_batch.grad is not None:
                x_step_batch.grad.zero_()

            # Backward pass
            if loss.item() > 0:  # Only backward if loss is non-zero
                loss.backward()
                if x_step_batch.grad is None:
                    print(f"Warning: No gradients for class {target_class_idx}, alpha {alpha:.2f}, loss={loss.item():.4f}, x_step_batch requires_grad={x_step_batch.requires_grad}")
                else:
                    attribution_values += x_step_batch.grad
                    print(f"Class {target_class_idx}, Alpha {alpha:.2f}, Loss: {loss.item():.4f}, Grad norm: {torch.norm(x_step_batch.grad):.4f}")
            else:
                print(f"Class {target_class_idx}, Alpha {alpha:.2f}, Loss: {loss.item():.4f}, Skipping backward due to zero loss")

        x_diff_mean = x_diff.mean(dim=0)
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        result = final_attribution.detach().cpu().numpy()
        if np.all(result == 0):
            print(f"Warning: All attribution values for class {target_class_idx} are zero. Check model output or baseline.")

        return result