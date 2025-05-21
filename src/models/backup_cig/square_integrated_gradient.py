import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor) 
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True)
    
    logits = model(inputs)
    if isinstance(logits, tuple):
        logits = logits[0]  # extract logits from (logits, Y_prob, Y_hat, _, instance_dict)
    
    if INPUT_OUTPUT_GRADIENTS in expected_keys:
        target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
        if logits.dim() == 2:
            target_output = logits[:, target_class_idx].sum()
        else:
            target_output = logits[target_class_idx].sum()
        gradients = torch.autograd.grad(target_output, inputs)[0]
        return {INPUT_OUTPUT_GRADIENTS: gradients}
    
    return logits

def normalize_by_2norm(x):
    batch_size = x.shape[0]
    norm = np.sqrt(np.sum(np.power(x, 2).reshape(batch_size, -1), axis=1))
    norm = np.where(norm == 0, 1e-8, norm)
    normed_x = np.moveaxis(x, 0, -1) / norm
    return np.moveaxis(normed_x, -1, 0)

class SquareIntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution (Square Form)"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value").to(kwargs.get("device", "cpu"))
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features").to(x_value.device)
        x_steps = kwargs.get("x_steps", 25)
        eta = kwargs.get("eta", 1)

        device = x_value.device
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        alphas = np.linspace(0, 1, x_steps)

        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices].squeeze(0)  # [N, D]
        x_diff = x_value - x_baseline_batch  # [N, D]

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing SIG", ncols=100), start=1):
            # Step 1: Compute model gradients w.r.t input
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = x_step_batch.clone().detach().requires_grad_(True).to(device)

            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)

            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS]  # np.ndarray
            gradients_avg = torch.tensor(gradients_batch, device=device, dtype=torch.float32).mean(dim=0)

            # Step 2: Compute counterfactual gradient via logits difference
            x_baseline_torch = x_baseline_batch.clone().detach().to(device)
            x_step_batch_cf = x_step_batch.clone().detach().requires_grad_(True).to(device)

            logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])
            logits_x_step = model(x_step_batch_cf, [x_step_batch_cf.shape[0]])

            # Extract actual tensors from CLAM's output tuple
            if isinstance(logits_x_r, tuple):
                logits_x_r = logits_x_r[0]
            if isinstance(logits_x_step, tuple):
                logits_x_step = logits_x_step[0]

            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            logits_difference.backward()

            if x_step_batch_cf.grad is None:
                raise RuntimeError("Gradients are not being computed!")

            grad_logits_diff = x_step_batch_cf.grad.detach()  # [N, D]
            counterfactual_gradients = grad_logits_diff.mean(dim=0)  # [D]

            # Step 3: Update attribution
            W_j = torch.norm(gradients_avg) + 1e-8
            contribution = (counterfactual_gradients * gradients_avg) * (eta / W_j)
            attribution_values += contribution

        return attribution_values.detach().cpu().numpy() / x_steps
