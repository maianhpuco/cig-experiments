import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs, call_model_function 

class EG(CoreSaliency):
    """Efficient Expected Gradients Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")                        # torch.Tensor
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")    # torch.Tensor
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cpu")

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        alphas = np.random.uniform(low=0.0, high=1.0, size=x_steps)

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            # Sample a baseline batch: [1, N, D] → [N, D]
            sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
            x_baseline_batch = baseline_features[sampled_indices].squeeze(0)  # shape: [N, D]
            x_diff = x_value - x_baseline_batch                                # shape: [N, D]

            # Interpolation step
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = x_step_batch.clone().detach().requires_grad_(True).to(device)

            # Forward + backward pass
            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)

            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS]  # shape: [N, D], np.ndarray
            gradients_avg = torch.tensor(gradients_batch, device=device)  # shape: [N, D]

            attribution_values += gradients_avg * x_diff  # shape: [N, D]

        return (attribution_values / x_steps).detach().cpu().numpy()
