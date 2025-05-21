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
        logits = logits[0]

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
    norm = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    norm = np.where(norm == 0, 1e-8, norm)
    return x / norm

class OptimSquareIntegratedGradients(CoreSaliency):
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
            x_step_tensor = torch.tensor(x_step, device=device, dtype=torch.float32)
            x_old_tensor = x_old.clone().detach().requires_grad_(True)

            call_model_output = call_model_function(
                x_old_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_old_tensor.shape, self.expected_keys)

            feature_gradient = torch.tensor(call_model_output[INPUT_OUTPUT_GRADIENTS], device=device).mean(dim=0)
            counterfactual_gradient = torch.tensor(counterfactual_gradients_memmap[i], device=device)

            W_j = torch.norm(feature_gradient) + 1e-8
            attr += (x_old - x_step_tensor) * feature_gradient * counterfactual_gradient * (eta / W_j)
            x_old = x_step_tensor

        return attr.detach().cpu().numpy()

    @staticmethod
    def Get_GradPath(x_value, baselines, model, x_steps, memmap_path, device):
        os.makedirs(memmap_path, exist_ok=True)
        path_filename = os.path.join(memmap_path, "grad_path_memmap.npy")
        counterfactual_grad_filename = os.path.join(memmap_path, "counterfactual_gradients_memmap.npy")

        path_shape = (x_steps, *x_value.shape)
        grad_shape = (x_steps - 1, *x_value.shape)

        path_memmap = np.memmap(path_filename, dtype=np.float32, mode='w+', shape=path_shape)
        counterfactual_grad_memmap = np.memmap(counterfactual_grad_filename, dtype=np.float32, mode='w+', shape=grad_shape)

        x_baseline_torch = baselines.clone().detach().to(device)
        logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])
        if isinstance(logits_x_r, tuple):
            logits_x_r = logits_x_r[0]

        delta = torch.zeros_like(x_value, device=device)
        path_memmap[0] = x_value.detach().cpu().numpy()

        step_size = 1.0
        prev_loss = float("inf")
        for i in tqdm(range(1, x_steps), desc="Searching GradPath", ncols=100):
            x_step = (x_value + delta).clone().detach().requires_grad_(True)
            logits_x_step = model(x_step, [x_step.shape[0]])
            if isinstance(logits_x_step, tuple):
                logits_x_step = logits_x_step[0]

            loss = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            grad = torch.autograd.grad(loss, x_step, retain_graph=True)[0]

            if grad is None:
                raise RuntimeError("Gradient is None; check model or requires_grad setting.")

            grad_np = normalize_by_2norm(grad.detach().cpu().numpy())
            counterfactual_grad_memmap[i - 1] = grad_np

            step_size *= 1.1 if loss.item() < prev_loss else 0.9
            prev_loss = loss.item()

            delta = delta + torch.tensor(grad_np, device=device, dtype=torch.float32) * step_size
            path_memmap[i] = (x_value + delta).detach().cpu().numpy()

            path_memmap.flush()
            counterfactual_grad_memmap.flush()
            torch.cuda.empty_cache()

        return path_memmap, counterfactual_grad_memmap
