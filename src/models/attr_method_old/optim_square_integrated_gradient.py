import torch
import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS

# Define missing constants manually
REP_LAYER_VALUES = "REP_LAYER_VALUES"
REP_DISTANCE_GRADIENTS = "REP_DISTANCE_GRADIENTS"

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    device = next(model.parameters()).device
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True).to(device)

    model.eval()
    outputs = model(inputs)
    logits_tensor = outputs[0] if isinstance(outputs, tuple) else outputs

    result = {}
    if expected_keys:
        if INPUT_OUTPUT_GRADIENTS in expected_keys:
            target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
            target_output = logits_tensor[:, target_class_idx].sum() if logits_tensor.dim() == 2 else logits_tensor[target_class_idx].sum()
            gradients = torch.autograd.grad(
                outputs=target_output,
                inputs=inputs,
                grad_outputs=torch.ones_like(target_output),
                create_graph=False,
                retain_graph=True
            )[0]
            result[INPUT_OUTPUT_GRADIENTS] = gradients.detach().cpu().numpy()

        if REP_DISTANCE_GRADIENTS in expected_keys and 'layer_baseline' in call_model_args:
            baseline_logits = call_model_args['layer_baseline']
            logits_diff = logits_tensor - baseline_logits
            loss = torch.norm(logits_diff, p=2) ** 2
            loss.backward()
            result[REP_DISTANCE_GRADIENTS] = inputs.grad.detach().cpu().numpy()
            inputs.grad.zero_()

        if REP_LAYER_VALUES in expected_keys:
            result[REP_LAYER_VALUES] = logits_tensor.detach().cpu().numpy()

        return result

    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    return logits_tensor[:, target_class_idx] if logits_tensor.dim() == 2 else logits_tensor

def normalize_by_2norm(x):
    norm = np.linalg.norm(x.reshape(x.shape[0], -1), axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-8, norm)
    return x / norm.reshape(x.shape[0], *([1] * (x.ndim - 1)))

class SquareIntegratedGradients(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def Get_GradPath(self, x_value, baselines, call_model_function, call_model_args=None, steps=50, step_size=1.0, clip_min_max=None):
        data = call_model_function(torch.tensor(baselines, dtype=torch.float32), call_model_args=call_model_args, expected_keys=[REP_LAYER_VALUES])
        call_model_args = call_model_args or {}
        call_model_args.update({'layer_baseline': torch.tensor(data[REP_LAYER_VALUES])})

        delta = np.zeros_like(x_value)
        path = [x_value.copy()]

        for _ in range(steps):
            step_input = x_value + delta
            data = call_model_function(torch.tensor(step_input, dtype=torch.float32), call_model_args=call_model_args, expected_keys=[REP_DISTANCE_GRADIENTS])
            grad = data[REP_DISTANCE_GRADIENTS]
            grad = normalize_by_2norm(grad)
            delta = delta + grad * step_size
            if clip_min_max:
                delta = np.clip(x_value + delta, clip_min_max[0], clip_min_max[1]) - x_value
            path.append((x_value + delta).copy())

        return np.array(path)

    def GetMask(self, **kwargs):
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        x_value = kwargs["x_value"]
        baseline_features = kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 50)
        step_size = kwargs.get("step_size", 1.0)
        clip_min_max = kwargs.get("clip_min_max", None)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        x_value = x_value.detach().cpu().numpy() if isinstance(x_value, torch.Tensor) else x_value
        baseline_features = baseline_features.detach().cpu().numpy() if isinstance(baseline_features, torch.Tensor) else baseline_features
        x_value = np.repeat(x_value[None, ...], baseline_features.shape[0], axis=0)

        path = self.Get_GradPath(x_value, baseline_features, call_model_function, call_model_args, x_steps, step_size, clip_min_max)
        np.testing.assert_allclose(x_value, path[0], rtol=0.01)

        attr = np.zeros_like(x_value, dtype=np.float32)
        x_old = x_value

        for x_step in path[1:]:
            x_old_tensor = torch.tensor(x_old, dtype=torch.float32, device=device)
            call_model_output = call_model_function(
                x_old_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=[INPUT_OUTPUT_GRADIENTS]
            )
            gradients = call_model_output[INPUT_OUTPUT_GRADIENTS]
            attr += (x_old - x_step) * gradients
            x_old = x_step

        return np.mean(attr, axis=0)

    def format_and_check_call_model_output(self, call_model_output, input_shape, expected_keys):
        pass
