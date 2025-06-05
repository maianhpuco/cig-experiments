import os
import numpy as np
import torch
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
import torch.nn.functional as F

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
            # Standardize input to prevent Tanh saturation
            x_step_batch = (baseline_features + alpha * x_diff).clone().detach()
            x_step_batch = (x_step_batch - x_step_batch.mean()) / (x_step_batch.std() + 1e-8)
            x_step_batch.requires_grad_(True).to(device)

            def debug_hook(grad):
                print(f"x_step_batch grad shape: {grad.shape}, norm: {torch.norm(grad)}")
            x_step_batch.register_hook(debug_hook)

            with torch.no_grad():
                logits_r = call_model_function(baseline_features, model, call_model_args)
                print(f"logits_r: {logits_r.detach().cpu().numpy()}")

            # Debug attention weights
            with torch.no_grad():
                call_model_args_temp = call_model_args.copy()
                call_model_args_temp["attention_only"] = True
                A = call_model_function(x_step_batch, model, call_model_args_temp)
                A_softmax = F.softmax(A, dim=1)
                print(f"Attention weights shape: {A.shape}, softmax norm: {torch.norm(A_softmax):.4f}, "
                      f"non-zero count: {(A_softmax > 1e-6).sum().item()}/{A_softmax.numel()}")

            logits_step = call_model_function(x_step_batch, model, call_model_args)
            print(f"logits_step: {logits_step.detach().cpu().numpy()}, requires_grad: {logits_step.requires_grad}")

            # Modified loss: L2 norm + class-specific term (mimicking IG)
            l2_loss = torch.norm(logits_step - logits_r, p=2) ** 2
            class_loss = logits_step[0, target_class_idx]  # Maximize target class logit
            loss = l2_loss - 0.5 * class_loss  # Negative to maximize
            print(f"Alpha {alpha:.2f}, Total Loss: {loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}, Class Loss: {class_loss.item():.4f}")

            # Gradient scaling
            loss = loss / (loss.item() + 1e-8)
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
            print("Warning: All attribution values are zero. Check attention weights or model output.")
        return result