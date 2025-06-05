import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

# class CIG(CoreSaliency):
#     """
#     Contrastive Gradients Attribution for per-class computation using autograd.
#     """
#     expected_keys = [INPUT_OUTPUT_GRADIENTS]

#     def GetMask(self, **kwargs):
#         x_value = kwargs["x_value"]
#         call_model_function = kwargs.get("call_model_function") or call_model_function 
#         model = kwargs["model"]
#         call_model_args = kwargs.get("call_model_args", {})
#         baseline_features = kwargs["baseline_features"]
#         x_steps = kwargs.get("x_steps", 25)
#         device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#         target_class_idx = call_model_args.get("target_class_idx", 0)

#         # Convert input types and move to device
#         x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device)
#         baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device)

#         attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

#         # Sample random baseline
#         try:
#             sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
#             x_baseline_batch = baseline_features[sampled_indices]
#         except Exception:
#             x_baseline_batch = torch.zeros_like(x_value, device=device)

#         x_diff = x_value - x_baseline_batch
#         if torch.norm(x_diff) < 1e-6:
#             x_baseline_batch = torch.zeros_like(x_value, device=device)
#             x_diff = x_value - x_baseline_batch

#         model.eval()
#         alphas = torch.linspace(0, 1, x_steps, device=device)

#         for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
#             x_step_batch = (x_baseline_batch + alpha * x_diff).clone().detach().requires_grad_(True).to(device)

#             with torch.no_grad():
#                 logits_r = call_model_function(x_baseline_batch, model, call_model_args)

#             call_model_output = call_model_function(
#                 x_step_batch,
#                 model,
#                 call_model_args=call_model_args,
#                 expected_keys=self.expected_keys
#             )

#             gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device, dtype=torch.float32)
#             attribution_values += gradients

#         x_diff_mean = x_diff.mean(dim=0)
#         final_attribution = (attribution_values * x_diff_mean) / x_steps
        
#         return final_attribution.detach().cpu().numpy()
# class CIG(CoreSaliency):
#     """
#     Contrastive Gradients Attribution for per-class computation.
#     """

#     def GetMask(self, **kwargs):
        
#         x_value = kwargs["x_value"]  # Expected: torch.Tensor [N, D]
#         model = kwargs["model"]
#         call_model_args = kwargs.get("call_model_args", {})
#         baseline_features = kwargs["baseline_features"]  # Expected: torch.Tensor [M, D]
#         x_steps = kwargs.get("x_steps", 25)
#         device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#         target_class_idx = call_model_args.get("target_class_idx", 0)
#         call_model_function = kwargs.get("call_model_function") or call_model_function  
#         # Ensure model parameters require gradients
#         for param in model.parameters():
#             if not param.requires_grad:
#                 print("Warning: Some model parameters are frozen, setting requires_grad=True")
#                 param.requires_grad_(True)

#         # Prepare tensors
#         x_value = (torch.tensor(x_value, dtype=torch.float32, device=device)
#                    if not isinstance(x_value, torch.Tensor) else x_value.to(device, dtype=torch.float32))
#         x_baseline_batch = (torch.tensor(baseline_features, dtype=torch.float32, device=device)
#                             if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device, dtype=torch.float32))

#         # Initialize attribution values
#         attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

#         # # Sample baseline indices
#         # try:
#         #     sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
#         #     x_baseline_batch = baseline_features[sampled_indices]  # [N, D]
#         # except (IndexError, ValueError):
#         #     print("Warning: Invalid baseline sampling, using zero baseline")
#         #     x_baseline_batch = torch.zeros_like(x_value, device=device)

#         x_diff = x_value - x_baseline_batch  # [N, D]
#         print("----shape of x_diff", x_diff)
#         # Check x_diff norm
#         x_diff_norm = torch.norm(x_diff).item()
#         print(f"x_diff norm: {x_diff_norm:.4f}")
#         if x_diff_norm < 1e-6:
#             print("Warning: x_diff is near zero, using zero baseline")
#             x_baseline_batch = torch.zeros_like(x_value, device=device)
#             x_diff = x_value - x_baseline_batch
#             x_diff_norm = torch.norm(x_diff).item()
#             print(f"New x_diff norm: {x_diff_norm:.4f}")

#         model.eval()
#         alphas = torch.linspace(0, 1, x_steps, device=device)

#         # Gradient hook to debug
#         def gradient_hook(module, grad_input, grad_output):
#             grad_out = grad_output[0] if isinstance(grad_output, tuple) else grad_output
#             print(f"Gradient hook: module={module.__class__.__name__}, name={module._get_name()}, grad_output shape={grad_out.shape if grad_out is not None else None}, grad_output requires_grad={grad_out.requires_grad if grad_out is not None else False}")

#         # Register hooks on classifier and attention layers
#         for name, module in model.named_modules():
#             if any(k in name.lower() for k in ["classifier", "fc", "attention", "attn"]):
#                 module.register_full_backward_hook(gradient_hook)
#                 print(f"Registered full backward hook on {name}")

#         for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
#             x_step_batch = x_baseline_batch + alpha * x_diff
#             x_step_batch = x_step_batch.clone().detach().requires_grad_(True).to(device)
#             x_baseline_torch = x_baseline_batch.clone().detach().to(device)

#             # Forward pass
#             logits_r = call_model_function(x_baseline_torch, model, call_model_args)
#             logits_step = call_model_function(x_step_batch, model, call_model_args)

#             # Check types and requires_grad
#             if not isinstance(logits_r, torch.Tensor) or not isinstance(logits_step, torch.Tensor):
#                 print(f"Error: Expected tensors, got logits_r: {type(logits_r)}, logits_step: {type(logits_step)}")
#                 raise TypeError("call_model_function returned non-tensor outputs")
#             if not logits_step.requires_grad:
#                 print(f"Error: logits_step does not require gradients for class {target_class_idx}, alpha {alpha:.2f}")
#                 raise RuntimeError("logits_step detached from computation graph")

#             logits_diff = (logits_step - logits_r).norm().item()
#             print(f"Class {target_class_idx}, Alpha {alpha:.2f}, logits_r shape: {logits_r.shape}, logits_step shape: {logits_r.shape}, logits_diff: {logits_diff:.4f}, logits_r: {logits_r.detach().cpu().numpy()}, logits_step: {logits_step.detach().cpu().numpy()}, logits_step requires_grad: {logits_step.requires_grad}")

#             # Compute counterfactual loss
#             loss = torch.norm(logits_step - logits_r, p=2) ** 2

#             # Zero gradients
#             if x_step_batch.grad is not None:
#                 x_step_batch.grad.zero_()

#             # Backward pass
#             if loss.item() > 0:  # Only backward if loss is non-zero
#                 loss.backward()
#                 if x_step_batch.grad is None:
#                     print(f"Warning: No gradients for class {target_class_idx}, alpha {alpha:.2f}, loss={loss.item():.4f}, x_step_batch requires_grad={x_step_batch.requires_grad}")
#                 else:
#                     attribution_values += x_step_batch.grad
#                     print(f"Class {target_class_idx}, Alpha {alpha:.2f}, Loss: {loss.item():.4f}, Grad norm: {torch.norm(x_step_batch.grad):.4f}")
#             else:
#                 print(f"Class {target_class_idx}, Alpha {alpha:.2f}, Loss: {loss.item():.4f}, Skipping backward due to zero loss")

#         x_diff_mean = x_diff.mean(dim=0)
#         final_attribution = (attribution_values * x_diff_mean) / x_steps

#         result = final_attribution.detach().cpu().numpy()
#         if np.all(result == 0):
#             print(f"Warning: All attribution values for class {target_class_idx} are zero. Check model output or baseline.")

#         return result 
 
class CIG(CoreSaliency):
    """
    Contrastive Gradients Attribution for per-class computation.
    """

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"]  # Expected: torch.Tensor [1, N, D]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"]  # Expected: torch.Tensor [N, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)
        call_model_function = kwargs.get("call_model_function", "") or call_model_function)

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

        # Verify shapes
        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        # Initialize attribution values
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        # Use baseline_features directly with batch dimension
        x_baseline_batch = baseline_features.unsqueeze(0)  # [1, N, D]
        print(f"x_baseline_batch shape: {x_value.shape_batch.shape}")
        with torch.no_grad():
            baseline_logits = call_model_function(x_baseline_batch, model, call_model_args)
            print(f"Baseline logits: {baseline_logits.detach().cpu().numpy()}")

        x_diff = x_value - x_baseline_batch  # [1, N, D]

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
        alphas = torch.linspace(0, 1, x_steps, device=device)[1:]  # Skip alpha=0

        for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
            x_step_batch = (x_baseline_batch + alpha * x_diff).clone().detach().requires_grad_(True).to(device)
            x_baseline_torch = x_baseline_batch.clone().detach().to(device)

            # Forward pass
            with torch.no_grad():
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

            # Compute gradients explicitly
            try:
                gradients = torch.autograd.grad(
                    outputs=loss,
                    inputs=x_step_batch,
                    grad_outputs=torch.ones_like(loss),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )[0]
            except RuntimeError as e:
                print(f"Gradient computation failed: {e}")
                continue

            if gradients is None:
                print(f"Warning: No gradients for class {target_class_idx}, alpha {alpha:.2f}, loss={loss.item():.4f}, x_step_batch requires_grad={x_step_batch.requires_grad}")
                continue

            attribution_values += gradients
            print(f"Class {target_class_idx}, Alpha {alpha:.2f}, Loss: {loss.item():.4f}, Grad norm: {torch.norm(gradients):.4f}")

        x_diff_mean = x_diff.mean(dim=1)  # Mean over patches: [1, D]
        final_attribution = (attribution_values * x_diff_mean) / x_steps

        result = final_attribution.detach().cpu().numpy()
        if np.all(result == 0):
            print(f"Warning: All attribution values for class {target_class_idx} are zero. Check model output or baseline.")

        return result