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
    if isinstance(outputs, tuple):
        logits_tensor = outputs[0]  # Assume first element is logits
        print(f"call_model_function: Unpacked tuple, logits shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}, is_leaf={logits_tensor.is_leaf}")
    else:
        logits_tensor = outputs
        print(f"call_model_function: Single output, logits shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}, is_leaf={logits_tensor.is_leaf}")

    if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0)
        if logits_tensor.dim() == 2:
            target_output = logits_tensor[:, target_class_idx].sum()
        else:
            target_output = logits_tensor[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        print(f"call_model_function: Computed gradients shape={gradients.shape}, requires_grad={gradients.requires_grad}")
        return {INPUT_OUTPUT_GRADIENTS: gradients}
    print(f"call_model_function: Returning raw outputs={type(outputs)}")
    return outputs

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

        # Use zero baseline for maximum contrast
        x_baseline_batch = torch.zeros_like(x_value, device=device)
        print(f"Zero baseline norm: {torch.norm(x_baseline_batch):.4f}")

        x_diff = x_value - x_baseline_batch

        if torch.norm(x_diff).item() < 1e-6:
            print("Warning: x_diff is near zero, using identity baseline")
            x_baseline_batch = x_value * 0.0
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

        # Test direct gradient link
        x_test = x_value.clone().requires_grad_(True)
        outputs_test = call_model_function(x_test, model, call_model_args)
        logits_test = outputs_test[0] if isinstance(outputs_test, tuple) else outputs_test
        target_logits_test = logits_test[:, target_class_idx].sum()
        grad_test = torch.autograd.grad(target_logits_test, x_test, allow_unused=True)[0]
        print(f"Direct gradient test: grad shape={grad_test.shape if grad_test is not None else None}, norm={torch.norm(grad_test) if grad_test is not None else 'None'}")

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
                outputs_r = call_model_function(x_baseline_batch, model, call_model_args)
                logits_r = outputs_r[0] if isinstance(outputs_r, tuple) else outputs_r
            outputs_step = call_model_function(x_step_tensor, model, call_model_args)
            logits_step = outputs_step[0] if isinstance(outputs_step, tuple) else outputs_step

            # Compute counterfactual gradient directly
            target_logits_step = logits_step[:, target_class_idx]
            counterfactual_grad = torch.autograd.grad(target_logits_step.sum(), x_step_tensor, allow_unused=True)[0]
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