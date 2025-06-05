# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

# s 
# class CIG(CoreSaliency):
#     """
#     Contrastive Integrated Gradients Attribution for per-class computation.
#     """

#     def GetMask(self, **kwargs):
#         x_value = kwargs["x_value"]  # Expected: torch.Tensor [1, N, D]
#         model = kwargs["model"]
#         call_model_args = kwargs.get("call_model_args", {})
#         baseline_features = kwargs["baseline_features"]  # Expected: torch.Tensor [N, D]
#         x_steps = kwargs.get("x_steps", 25)
#         device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#         target_class_idx = call_model_args.get("target_class_idx", 0)
#         call_model_function = kwargs["call_model_function"]

#         # Ensure model parameters require gradients
#         for param in model.parameters():
#             param.requires_grad_(True)

#         # Prepare input tensors
#         x_value = torch.tensor(x_value, dtype=torch.float32, device=device) if not isinstance(x_value, torch.Tensor) else x_value.to(device, dtype=torch.float32)
#         baseline_features = torch.tensor(baseline_features, dtype=torch.float32, device=device) if not isinstance(baseline_features, torch.Tensor) else baseline_features.to(device, dtype=torch.float32)

#         # Check dimensions
#         if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
#             raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

#         # Initialize attribution tensor
#         attribution_values = torch.zeros_like(x_value, device=device)

#         # Expand baseline to match input shape
#         x_baseline_batch = baseline_features.unsqueeze(0)  # Shape: [1, N, D]
#         x_diff = x_value - x_baseline_batch  # Shape: [1, N, D]

#         if torch.norm(x_diff).item() < 1e-6:
#             print("x_diff near zero, using zero baseline")
#             x_baseline_batch = torch.zeros_like(x_value)
#             x_diff = x_value - x_baseline_batch

#         model.eval()
#         alphas = torch.linspace(0, 1, x_steps, device=device)[1:]  # skip alpha=0

#         for alpha in tqdm(alphas, desc=f"Computing class {target_class_idx}:", ncols=100):
#             x_step_batch = x_baseline_batch + alpha * x_diff
#             x_baseline_batch = torch.tensor(x_baseline_batch.copy(), dtype=torch.float32, requires_grad=False)
#             # Get baseline logits without tracking gradients
#             # with torch.no_grad():
#             x_step_batch_torch = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True) 
            
#             logits_r = call_model_function(x_baseline_batch, model, call_model_args)
#             if isinstance(logits_r, tuple):
#                 logits_r = logits_r[0]

#             # Forward pass with gradient tracking
#             logits_step = call_model_function(x_step_batch_torch, model, call_model_args)
            
#             if isinstance(logits_step, tuple):
#                 logits_step = logits_step[0]
                
#             if x_step_batch_torch.grad is None:
#                 raise RuntimeError("-> x_step_batch Gradients are not being computed! Ensure tensors require gradients.")
 
#             if not logits_step.requires_grad:
#                 raise RuntimeError("logits_step does not require gradients")

#             # Compute contrastive loss
#             loss = torch.norm(logits_step - logits_r, p=2) ** 2
#             print(f"[Debug] logits_step.requires_grad: {logits_step.requires_grad}")
#             loss.backward()
            
#             if logits_step.grad is None:
#                 raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.") 
#             # Compute gradients
#             gradients = torch.autograd.grad(
#                 outputs=loss,
#                 inputs=x_step_batch,
#                 grad_outputs=torch.ones_like(loss),
#                 retain_graph=False,
#                 create_graph=False,
#                 allow_unused=True
#             )[0]

#             if gradients is not None:
#                 attribution_values += gradients
#             else:
#                 print(f"No gradients at alpha {alpha:.2f}, skipping")

#         x_diff_mean = x_diff.mean(dim=1, keepdim=True)  # Shape: [1, 1, D]
#         final_attribution = (attribution_values * x_diff_mean) / x_steps

#         result = final_attribution.detach().cpu().numpy()

#         if np.all(result == 0):
#             print(f"Warning: All attribution values for class {target_class_idx} are zero.")

#         return result


import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

def call_model_function(inputs, model, call_model_args=None):
    device = next(model.parameters()).device
    model.eval()

    # Ensure input has batch dimension
    if inputs.dim() == 2:  # [N, D] -> [1, N, D]
        inputs = inputs.unsqueeze(0)
    elif inputs.dim() != 3:
        raise ValueError(f"Expected inputs to be 2D or 3D, got shape {inputs.shape}")

    # Ensure input requires grad
    inputs = inputs.to(device, dtype=torch.float32)
    inputs.requires_grad_(True)

    # Run model
    outputs = model(inputs)

    # Handle models returning tuples or dicts
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    elif isinstance(outputs, dict):
        outputs = outputs.get("logits", list(outputs.values())[0])

    print(f"call_model_function: inputs shape={inputs.shape}, requires_grad={inputs.requires_grad}, outputs shape={outputs.shape}, requires_grad={outputs.requires_grad}")
    return outputs

# class CIG(CoreSaliency):
#     """
#     Contrastive Integrated Gradients Attribution for per-class computation.
#     """

#     def GetMask(self, **kwargs):
#         x_value = kwargs.get("x_value")  # Expected: torch.Tensor [1, N, D]
#         model = kwargs.get("model")
#         call_model_args = kwargs.get("call_model_args", {})
#         baseline_features = kwargs.get("baseline_features")  # Expected: torch.Tensor [N, D]
#         x_steps = kwargs.get("x_steps", 25)
#         device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#         target_class_idx = call_model_args.get("target_class_idx", 0)
#         call_model_function = kwargs.get("call_model_function") or call_model_function

#         # Ensure inputs are torch tensors on correct device
#         x_value = x_value.detach().clone().to(device, dtype=torch.float32)
#         baseline_features = baseline_features.detach().clone().to(device, dtype=torch.float32)

#         # Validate shapes
#         if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
#             raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

#         # Add batch dimension to baseline
#         baseline_features = baseline_features.unsqueeze(0)  # [1, N, D]
#         print(f"baseline_features shape: {baseline_features.shape}, x_value shape: {x_value.shape}")

#         attribution_values = torch.zeros_like(x_value, device=device)

#         x_diff = x_value - baseline_features  # [1, N, D]
#         x_diff_norm = torch.norm(x_diff).item()
#         print(f"x_diff norm: {x_diff_norm:.4f}")
#         if x_diff_norm < 1e-6:
#             print("Warning: x_diff is near zero, using zero baseline")
#             baseline_features = torch.zeros_like(x_value, device=device)
#             x_diff = x_value - baseline_features

#         alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # Skip alpha=0

#         for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
#             x_step_batch = (baseline_features + alpha * x_diff).clone().detach().requires_grad_(True).to(device)

#             with torch.no_grad():
#                 logits_r = call_model_function(baseline_features, model, call_model_args)
#                 if isinstance(logits_r, dict):
#                     logits_r = logits_r.get("logits", list(logits_r.values())[0])
#                 if isinstance(logits_r, tuple):
#                     logits_r = logits_r[0]
#                 print(f"logits_r: {logits_r.detach().cpu().numpy()}")

#             logits_step = call_model_function(x_step_batch, model, call_model_args)
#             if isinstance(logits_step, dict):
#                 logits_step = logits_step.get("logits", list(logits_step.values())[0])
#             if isinstance(logits_step, tuple):
#                 logits_step = logits_step[0]
#             print(f"logits_step: {logits_step.detach().cpu().numpy()}, requires_grad: {logits_step.requires_grad}")

#             loss = torch.norm(logits_step - logits_r, p=2) ** 2
#             print(f"Alpha {alpha:.2f}, Loss: {loss.item():.4f}")

#             try:
#                 gradients = torch.autograd.grad(
#                     outputs=loss,
#                     inputs=x_step_batch,
#                     grad_outputs=torch.ones_like(loss),
#                     retain_graph=False,
#                     create_graph=False,
#                     allow_unused=True
#                 )[0]
#             except RuntimeError as e:
#                 print(f"Gradient computation failed at alpha {alpha:.2f}: {e}")
#                 continue

#             if gradients is None:
#                 print(f"No gradients at alpha {alpha:.2f}, loss={loss.item():.4f}, x_step_batch requires_grad={x_step_batch.requires_grad}")
#                 continue

#             # Mean over batch dimension if present
#             counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
#             attribution_values += counterfactual_gradients

#         x_diff_mean = x_diff.mean(dim=0)  # [N, D]
#         attribution_values *= x_diff_mean

#         result = attribution_values.detach().cpu().numpy() / x_steps
#         if np.all(result == 0):
#             print("Warning: All attribution values are zero. Check model output or baseline.")
#         return result
def call_model_function(inputs, model, call_model_args=None):
    device = next(model.parameters()).device
    model.eval()

    if inputs.dim() == 2:  # [N, D] -> [1, N, D]
        inputs = inputs.unsqueeze(0)
    elif inputs.dim() != 3:
        raise ValueError(f"Expected inputs to be 2D or 3D, got shape {inputs.shape}")

    inputs = inputs.to(device, dtype=torch.float32)
    inputs.requires_grad_(True)

    outputs = model(inputs)  # CLAM returns (logits, Y_prob, Y_hat, A_raw, results_dict)
    logits = outputs[0]  # Extract logits

    print(f"call_model_function: inputs shape={inputs.shape}, requires_grad={inputs.requires_grad}, "
          f"logits shape={logits.shape}, requires_grad={logits.requires_grad}")
    return logits

class CIG(CoreSaliency):
    """
    Contrastive Integrated Gradients Attribution for per-class computation.
    """

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # Expected: torch.Tensor [1, N, D]
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features")  # Expected: torch.Tensor [N, D]
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)
        call_model_function = kwargs.get("call_model_function") or call_model_function

        x_value = x_value.detach().clone().to(device, dtype=torch.float32)
        baseline_features = baseline_features.detach().clone().to(device, dtype=torch.float32)

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        baseline_features = baseline_features.unsqueeze(0)  # [1, N, D]
        print(f"baseline_features shape: {baseline_features.shape}, x_value shape: {x_value.shape}")

        attribution_values = torch.zeros_like(x_value, device=device)

        x_diff = x_value - baseline_features
        x_diff_norm = torch.norm(x_diff).item()
        print(f"x_diff norm: {x_diff_norm:.4f}")
        if x_diff_norm < 1e-6:
            print("Warning: x_diff is near zero, using zero baseline")
            baseline_features = torch.zeros_like(x_value, device=device)
            x_diff = x_value - baseline_features

        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = (baseline_features + alpha * x_diff).clone().detach().requires_grad_(True).to(device)

            def debug_hook(grad):
                print(f"x_step_batch grad shape: {grad.shape}, norm: {torch.norm(grad)}")
            x_step_batch.register_hook(debug_hook)

            with torch.no_grad():
                logits_r = call_model_function(baseline_features, model, call_model_args)
                print(f"logits_r: {logits_r.detach().cpu().numpy()}")

            # Compute attention weights for debugging
            # with torch.no_grad():
            A = model(x_step_batch, attention_only=True)  # [1, N] or [n_classes, N]
            print(f"Attention weights shape: {A.shape}, norm: {torch.norm(A):.4f}, max: {A.max():.4f}, min: {A.min():.4f}")

            logits_step = call_model_function(x_step_batch, model, call_model_args)
            print(f"logits_step: {logits_step.detach().cpu().numpy()}, requires_grad: {logits_step.requires_grad}")

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            print(f"Alpha {alpha:.2f}, Loss: {loss.item():.4f}")

            # Gradient scaling to prevent vanishing
            loss = loss / (loss.item() + 1e-8)  # Normalize loss
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
                print(f"Gradient computation failed at alpha {alpha:.2f}: {e}")
                continue

            if gradients is None:
                print(f"No gradients at alpha {alpha:.2f}, loss={loss.item():.4f}, x_step_batch requires_grad={x_step_batch.requires_grad}")
                continue

            counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += counterfactual_gradients
            print(f"Gradients shape: {counterfactual_gradients.shape}, norm: {torch.norm(counterfactual_gradients):.4f}")

        x_diff_mean = x_diff.mean(dim=0)
        attribution_values *= x_diff_mean

        result = attribution_values.detach().cpu().numpy() / x_steps
        if np.all(result == 0):
            print("Warning: All attribution values are zero. Check model output or baseline.")
        return result 