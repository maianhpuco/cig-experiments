import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
# from attr_method._common import PreprocessInputs, call_model_function 


def normalize_by_2norm(x):
    """Normalize gradients using L2 norm."""
    batch_size = x.shape[0]
    norm = np.sqrt(np.sum(np.power(x, 2).reshape(batch_size, -1), axis=1))  # L2 norm
    norm = np.where(norm == 0, 1e-8, norm)  # Avoid division by zero
    normed_x = np.moveaxis(x, 0, -1) / norm
    return np.moveaxis(normed_x, -1, 0) 


class OptimSquareIntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with GradPath (IG²)"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value").to(kwargs.get("device", "cpu"))
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features").to(x_value.device)
        x_steps = kwargs.get("x_steps", 5)
        eta = kwargs.get("eta", 1)
        memmap_path = kwargs.get("memmap_path")

        device = x_value.device
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices].squeeze(0)

        path, counterfactual_gradients_memmap = self.Get_GradPath(x_value, x_baseline_batch, model, x_steps, memmap_path, device)

        np.testing.assert_allclose(x_value.cpu().numpy(), path[0], rtol=0.01)

        print('Integrating gradients on GradPath...')
        attr = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        x_old = x_value.clone()

        for i, x_step in enumerate(path[1:]):
            x_old_tensor = x_old.clone().detach().requires_grad_(True).to(device)

            call_model_output = call_model_function(
                x_old_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_old_tensor.shape, self.expected_keys)

            feature_gradient = torch.tensor(
                call_model_output[INPUT_OUTPUT_GRADIENTS],
                dtype=torch.float32,
                device=device
            ).mean(dim=0)  # shape: [N, D]

            counterfactual_gradient = torch.tensor(counterfactual_gradients_memmap[i], device=device)

            W_j = torch.norm(feature_gradient) + 1e-8
            attr += (x_old - torch.tensor(x_step, device=device)) * feature_gradient * counterfactual_gradient * (eta / W_j)
            x_old = torch.tensor(x_step, device=device)

        return attr.detach().cpu().numpy()

    @staticmethod
    def Get_GradPath(x_value, baselines, model, x_steps, memmap_path, device):
        path_filename = f"{memmap_path}/grad_path_memmap.npy"
        counterfactual_grad_filename = f"{memmap_path}/counterfactual_gradients_memmap.npy"
        path_shape = (x_steps, *x_value.shape)
        grad_shape = (x_steps - 1, *x_value.shape)

        path_memmap = np.memmap(path_filename, dtype=np.float32, mode='w+', shape=path_shape)
        counterfactual_gradients_memmap = np.memmap(counterfactual_grad_filename, dtype=np.float32, mode='w+', shape=grad_shape)

        x_baseline_torch = baselines.clone().detach().to(device)
        logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])

        delta = torch.zeros_like(x_value, device=device)
        path_memmap[0] = x_value.detach().cpu().numpy()

        progress_bar = tqdm(range(1, x_steps), desc="Searching GradPath", ncols=100)
        step_size = 1.0
        prev_loss = float('inf')

        for i in progress_bar:
            x_step_batch = (x_value + delta).clone().detach().requires_grad_(True)

            logits_x_step = model(x_step_batch, [x_step_batch.shape[0]])
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2

            grad_logits_diff = torch.autograd.grad(logits_difference, x_step_batch, retain_graph=True)[0]

            if grad_logits_diff is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff_np = grad_logits_diff.detach().cpu().numpy()
            grad_logits_diff_np = normalize_by_2norm(grad_logits_diff_np)

            counterfactual_gradients_memmap[i - 1] = grad_logits_diff_np

            if logits_difference.item() < prev_loss:
                step_size *= 1.1
            else:
                step_size *= 0.9
            prev_loss = logits_difference.item()

            delta = delta + torch.tensor(grad_logits_diff_np, device=device) * step_size
            x_adv = x_value + delta

            path_memmap[i] = x_adv.detach().cpu().numpy()
            path_memmap.flush()
            counterfactual_gradients_memmap.flush()

            progress_bar.set_postfix({"Loss": logits_difference.item(), "Step Size": step_size})
            torch.cuda.empty_cache()

        return path_memmap, counterfactual_gradients_memmap
