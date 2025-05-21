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
        return {INPUT_OUTPUT_GRADIENTS: gradients}
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
        x_steps = kwargs.get("x_steps", 25)
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
            x_baseline_batch = torch.zeros_like(x_value)

        x_diff = x_value - x_baseline_batch

        if torch.norm(x_diff).item() < 1e-6:
            x_baseline_batch = torch.zeros_like(x_value)
            x_diff = x_value - x_baseline_batch

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

            # Step 2: Counterfactual gradients
            with torch.no_grad():
                logits_r = call_model_function(x_baseline_batch, model, call_model_args)
            logits_step = call_model_function(x_step_tensor, model, call_model_args)

            logits_diff = (logits_step - logits_r).norm(p=2) ** 2
            logits_diff.backward()

            if x_step_tensor.grad is None:
                raise RuntimeError("No gradient found for x_step_tensor")

            grad_logits_diff = x_step_tensor.grad
            counterfactual_grad = grad_logits_diff.mean(dim=0)

            W_j = torch.norm(gradients_avg) + 1e-8
            attribution_values += (counterfactual_grad * gradients_avg) * (eta / W_j)

        return (attribution_values / x_steps).detach().cpu().numpy()
