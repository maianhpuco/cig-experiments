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
    print(f"call_model_function: logits_tensor shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}")

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
    Efficient Integrated Gradients Squared (IG²) with CLAM model compatibility.
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        # Convert x_value to NumPy for consistency
        x_value = kwargs["x_value"].detach().cpu().numpy() if isinstance(kwargs["x_value"], torch.Tensor) else kwargs["x_value"]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"].detach().cpu().numpy() if isinstance(kwargs["baseline_features"], torch.Tensor) else kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 25)
        eta = kwargs.get("eta", 1.0)  # Kept for compatibility, though not used in IG²
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Convert to torch tensors and move to device
        x_value = torch.tensor(x_value, dtype=torch.float32, device=device)
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        # Sample random baseline
        try:
            sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
            x_baseline_batch = torch.tensor(baseline_features[sampled_indices], dtype=torch.float32, device=device)
        except Exception as e:
            print(f"Warning: Failed to sample baseline ({e}), using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch
        if torch.norm(x_diff) < 1e-6:
            print("Warning: x_diff is near zero, using identity baseline")
            x_baseline_batch = torch.zeros_like(x_value)
            x_diff = x_value - x_baseline_batch

        print(f"x_value shape: {x_value.shape}, norm: {torch.norm(x_value):.4f}")
        print(f"x_diff shape: {x_diff.shape}, norm: {torch.norm(x_diff):.4f}")

        # Forward hook to debug requires_grad
        def forward_hook(module, input, output):
            output_tensor = output[0] if isinstance(output, tuple) else output
            print(f"Forward hook: module={module.__class__.__name__}, input shape={input[0].shape if isinstance(input, tuple) else input.shape}, output shape={output_tensor.shape}, requires_grad={output_tensor.requires_grad}")

        # Register hooks on key layers
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["classifier", "fc", "attention", "attn"]):
                module.register_forward_hook(forward_hook)

        # Linear path from baseline to x_value
        alphas = torch.linspace(0, 1, x_steps, device=device)
        x_old = x_value

        for alpha in tqdm(alphas, desc=f"IG² class {target_class_idx}", ncols=100):
            x_step = x_baseline_batch + alpha * x_diff
            x_step_tensor = x_step.clone().detach().requires_grad_(True)
            print(f"Input x_step_tensor shape: {x_step_tensor.shape}")

            # Compute gradient at x_old
            call_model_output = call_model_function(
                x_old,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            gradients = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device, dtype=torch.float32)
            print(f"Alpha {alpha:.2f}, gradients shape: {gradients.shape}, norm: {torch.norm(gradients):.4f}, min: {gradients.min():.4e}, max: {gradients.max():.4e}")

            # IG² attribution update: (x_old - x_step) * gradient
            attribution_values += (x_old - x_step) * gradients

            x_old = x_step

        # Final attribution (average over steps)
        final_attribution = attribution_values / x_steps
        result = final_attribution.detach().cpu().numpy()
        print(f"Raw attribution norm: {np.linalg.norm(result):.4f}, min: {result.min():.4e}, max: {result.max():.4e}")
        return result