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

class CIG(CoreSaliency):
    """
    Contrastive Integrated Gradients Attribution for per-class computation.
    """

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features")
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)
        call_model_function = kwargs.get("call_model_function")

        # Convert to numpy if input is tensor
        if isinstance(x_value, torch.Tensor):
            x_value = x_value.detach().cpu().numpy()
        if isinstance(baseline_features, torch.Tensor):
            baseline_features = baseline_features.detach().cpu().numpy()

        attribution_values =  np.zeros_like(x_value, dtype=np.float32)
        # total_grad =  np.zeros_like(x_value, dtype=np.float32)
         
        alphas = np.linspace(0, 1, x_steps)
        # sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[-2]), replace=True)
        x_baseline_batch = baseline_features.unsqueeze(0)
        x_diff = x_value - x_baseline_batch 
        
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            # ------------ Counter Factual Gradient ------------ 
            x_step_batch = x_baseline_batch + alpha * x_diff
            # ------------ Counter Factual Gradient ------------
            x_baseline_torch = torch.tensor(x_baseline_batch.copy(), dtype=torch.float32, requires_grad=False)
            print("x_baseline_torch", x_baseline_torch.shape)
            logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])

            # Compute counterfactual gradients using logitws difference
            x_step_batch_torch = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)
            logits_x_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            logits_difference.backward()
            
            if x_step_batch_torch.grad is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff = x_step_batch_torch.grad.numpy()
            
            # ------------ Conbine Gradient and X_diff ------------ 
            counterfactual_gradients = grad_logits_diff.mean(axis=0) 
            
            # print("check shape")
            # print(x_diff.shape, counterfactual_gradients.shape)
            attribution_values += counterfactual_gradients 
            
        x_diff = x_diff.mean(axis=0) 
        
        attribution_values = attribution_values * x_diff 
        
        return attribution_values / x_steps 