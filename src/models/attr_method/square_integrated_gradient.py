import torch
import numpy as np
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
        return {INPUT_OUTPUT_GRADIENTS: gradients.detach().cpu().numpy()}

    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    if logits_tensor.dim() == 2:
        return logits_tensor[:, target_class_idx]
    return logits_tensor

class SquareIntegratedGradients(CoreSaliency):
    """
    Efficient Integrated Gradients with Counterfactual Attribution (Square Form)
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        # Convert x_value to NumPy for consistency
        x_value = kwargs["x_value"].detach().cpu().numpy() if isinstance(kwargs["x_value"], torch.Tensor) else kwargs["x_value"]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"].detach().cpu().numpy() if isinstance(kwargs["baseline_features"], torch.Tensor) else kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 25)  # Match original code's default
        eta = kwargs.get("eta", 1.0)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Prepare attribution accumulator
        attribution_values = np.zeros_like(x_value, dtype=np.float32)
        alphas = np.linspace(0, 1, x_steps)

        # Sample baseline without extra batch dimension
        try:
            sampled_indices = np.random.choice(baseline_features.shape[0], x_value.shape[0], replace=True)
            x_baseline_batch = baseline_features[sampled_indices]
            print(f"Sampled baseline shape: {x_baseline_batch.shape}, norm: {np.linalg.norm(x_baseline_batch):.4f}, min: {x_baseline_batch.min():.4f}, max: {x_baseline_batch.max():.4f}")
        except Exception as e:
            print(f"Warning: Failed to sample baseline ({e}), using zero baseline")
            x_baseline_batch = np.zeros_like(x_value, dtype=np.float32)

        x_diff = x_value - x_baseline_batch

        if np.linalg.norm(x_diff) < 1e-6:
            print("Warning: x_diff is near zero, using identity baseline")
            x_baseline_batch = np.zeros_like(x_value)
            x_diff = x_value - x_baseline_batch

        print(f"x_value shape: {x_value.shape}, norm: {np.linalg.norm(x_value):.4f}")
        print(f"x_diff shape: {x_diff.shape}, norm: {np.linalg.norm(x_diff):.4f}")

        # Forward hook to debug requires_grad
        def forward_hook(module, input, output):
            output_tensor = output[0] if isinstance(output, tuple) else output
            print(f"Forward hook: module={module.__class__.__name__}, input shape={input[0].shape if isinstance(input, tuple) else input.shape}, output shape={output_tensor.shape}, requires_grad={output_tensor.requires_grad}")

        # Register hooks on key layers
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["classifier", "fc", "attention", "attn"]):
                module.register_forward_hook(forward_hook)

        for alpha in tqdm(alphas, desc=f"SIG class {target_class_idx}", ncols=100):
            x_step = x_baseline_batch + alpha * x_diff
            x_step_tensor = torch.tensor(x_step, dtype=torch.float32, device=device, requires_grad=True)
            print(f"Input x_step_tensor shape: {x_step_tensor.shape}")

            # Step 1: Get model gradient w.r.t x_step
            call_model_output = call_model_function(
                x_step_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            gradients = call_model_output[INPUT_OUTPUT_GRADIENTS]
            gradients_avg = gradients.mean(axis=0)
            print(f"Alpha {alpha:.2f}, gradients_avg shape: {gradients_avg.shape}, norm: {np.linalg.norm(gradients_avg):.4f}")

            # Step 2: Counterfactual gradients using .backward()
            with torch.no_grad():
                x_baseline_torch = torch.tensor(x_baseline_batch, dtype=torch.float32, device=device)
                logits_r = call_model_function(x_baseline_torch, model, call_model_args)
                logits_r = logits_r[0] if isinstance(logits_r, tuple) else logits_r

            x_step_tensor = torch.tensor(x_step, dtype=torch.float32, device=device, requires_grad=True)
            logits_step = call_model_function(x_step_tensor, model, call_model_args)
            logits_step = logits_step[0] if isinstance(logits_step, tuple) else logits_step

            logits_diff = torch.norm(logits_step - logits_r, p=2) ** 2
            logits_diff.backward()

            if x_step_tensor.grad is None:
                print(f"Warning: x_step_tensor.grad is None for alpha={alpha:.2f}, using gradients_avg as fallback")
                counterfactual_grad = gradients_avg
            else:
                counterfactual_grad = x_step_tensor.grad.mean(dim=0).detach().cpu().numpy()
                print(f"Alpha {alpha:.2f}, counterfactual_grad shape: {counterfactual_grad.shape}, norm: {np.linalg.norm(counterfactual_grad):.4f}")

            W_j = np.linalg.norm(gradients_avg) + 1e-8
            attribution_values += (counterfactual_grad * gradients_avg) * (eta / W_j)

        # Final attribution
        result = attribution_values / x_steps
        print(f"Raw attribution norm: {np.linalg.norm(result):.4f}, min: {result.min():.4e}, max: {result.max():.4e}")
        return result