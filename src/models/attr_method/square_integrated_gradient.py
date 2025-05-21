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
    outputs = model(inputs)
    logits_tensor = outputs[0] if isinstance(outputs, tuple) else outputs

    if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0)
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        print(f"call_model_function: Computed gradients shape={gradients.shape}, requires_grad={gradients.requires_grad}")
        return {INPUT_OUTPUT_GRADIENTS: gradients}
    print(f"call_model_function: Returning logits shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}")
    return logits_tensor

class SquareIntegratedGradients(CoreSaliency):
    """
    Efficient Integrated Gradients with Counterfactual Attribution (Square Form)
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        x_value = kwargs["x_value"].to(kwargs.get("device", "cpu"))
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"].to(x_value.device)
        x_steps = kwargs.get("x_steps", 50)
        eta = kwargs.get("eta", 1.0)
        device = x_value.device
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Prepare attribution accumulator
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        alphas = torch.linspace(0, 1, x_steps, device=device)

        # Sample baseline
        try:
            sampled_indices = torch.randint(0, baseline_features.shape[0], (x_value.shape[0],), device=device)
            x_baseline_batch = baseline_features[sampled_indices]
        except Exception:
            print("Warning: Invalid baseline sampling, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value, device=device)

        x_diff = x_value - x_baseline_batch

        if torch.norm(x_diff).item() < 1e-6:
            print("Warning: x_diff is near zero, using zero baseline")
            x_baseline_batch = torch.zeros_like(x_value)
            x_diff = x_value - x_baseline_batch

        print(f"x_value shape: {x_value.shape}, norm: {torch.norm(x_value):.4f}")
        print(f"x_diff shape: {x_diff.shape}, norm: {torch.norm(x_diff):.4f}")

        # Forward hook to debug requires_grad
        def forward_hook(module, input, output):
            output_tensor = output[0] if isinstance(output, tuple) else output
            print(f"Forward hook: module={module.__class__.__name__}, output shape={output_tensor.shape}, requires_grad={output_tensor.requires_grad}")

        # Register hooks on key layers
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["classifier", "fc", "attention", "attn"]):
                module.register_forward_hook(forward_hook)

        for alpha in tqdm(alphas, desc=f"SIG class {target_class_idx}", ncols=100):
            x_step = x_baseline_batch + alpha * x_diff
            x_step_tensor = x_step.clone().detach().requires_grad_(True)

            # Step 1: Get model gradient w.r.t x_step
            call_model_output = call_model_function(
                x_step_tensor, model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            gradients = call_model_output[INPUT_OUTPUT_GRADIENTS]
            gradients_avg = gradients.mean(dim=0)
            print(f"Alpha {alpha:.2f}, gradients_avg shape: {gradients_avg.shape}, norm: {torch.norm(gradients_avg):.4f}")

            # Step 2: Counterfactual gradients
            with torch.no_grad():
                logits_r = call_model_function(x_baseline_batch, model, call_model_args)
            logits_step = call_model_function(x_step_tensor, model, call_model_args)

            # Compute logits difference for the target class only
            target_logits_step = logits_step[:, target_class_idx]
            target_logits_r = logits_r[:, target_class_idx]
            logits_diff = (target_logits_step - target_logits_r).norm(p=2) ** 2

            # Compute counterfactual gradient with allow_unused=True
            grad_outputs = torch.ones_like(logits_diff)
            counterfactual_grad = torch.autograd.grad(logits_diff, x_step_tensor, grad_outputs=grad_outputs, allow_unused=True)[0]
            if counterfactual_grad is None:
                print(f"Warning: counterfactual_grad is None for alpha={alpha:.2f}, using zeros")
                counterfactual_grad = torch.zeros_like(gradients_avg)
            else:
                counterfactual_grad = counterfactual_grad.mean(dim=0)
                print(f"Alpha {alpha:.2f}, counterfactual_grad shape: {counterfactual_grad.shape}, norm: {torch.norm(counterfactual_grad):.4f}")

            W_j = torch.norm(gradients_avg) + 1e-8
            attribution_values += (counterfactual_grad * gradients_avg) * (eta / W_j)

        # Final attribution
        result = (attribution_values / x_steps).detach().cpu().numpy()
        print(f"Raw attribution norm: {torch.norm(attribution_values):.4f}, min: {result.min():.4e}, max: {result.max():.4e}")
        return result