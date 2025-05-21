import torch
import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS, REP_LAYER_VALUES, REP_DISTANCE_GRADIENTS

def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
    """
    Generic model call with gradient support for class attribution, extended for IG².
    """
    device = next(model.parameters()).device
    inputs = (inputs.clone().detach() if isinstance(inputs, torch.Tensor)
              else torch.tensor(inputs, dtype=torch.float32)).requires_grad_(True).to(device)

    model.eval()
    outputs = model(inputs)  # CLAM returns (logits, prob, pred, _, dict)
    logits_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
    print(f"call_model_function: logits_tensor shape={logits_tensor.shape}, requires_grad={logits_tensor.requires_grad}")

    if expected_keys:
        result = {}
        if INPUT_OUTPUT_GRADIENTS in expected_keys:
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
                retain_graph=True
            )[0]
            result[INPUT_OUTPUT_GRADIENTS] = gradients.detach().cpu().numpy()
        
        if REP_DISTANCE_GRADIENTS in expected_keys and 'layer_baseline' in call_model_args:
            baseline_logits = call_model_args['layer_baseline']
            logits_diff = logits_tensor - baseline_logits
            loss = torch.norm(logits_diff, p=2) ** 2
            loss.backward()
            result[REP_DISTANCE_GRADIENTS] = inputs.grad.detach().cpu().numpy()
            inputs.grad.zero_()  # Clear gradients to avoid accumulation
        
        if REP_LAYER_VALUES in expected_keys:
            result[REP_LAYER_VALUES] = logits_tensor.detach().cpu().numpy()
        
        if result:
            return result

    target_class_idx = call_model_args.get('target_class_idx', 0) if call_model_args else 0
    if logits_tensor.dim() == 2:
        return logits_tensor[:, target_class_idx]
    return logits_tensor

def normalize_by_2norm(x):
    """
    Normalize input by 2-norm along the batch dimension.
    """
    batch_size = x.shape[0]
    norm = np.power(np.sum(np.power(np.abs(x), 2).reshape(batch_size, -1), axis=1), 1.0 / 2)
    norm = np.where(norm == 0, 1e-8, norm)  # Avoid division by zero
    normed_x = np.moveaxis(x, 0, -1) / norm
    return np.moveaxis(normed_x, -1, 0)

class SquareIntegratedGradients(CoreSaliency):
    """
    Integrated Gradients Squared (IG²) with iterative gradient path optimization.
    """
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def Get_GradPath(self, x_value, baselines, call_model_function, call_model_args=None, steps=201, step_size=1.0, clip_min_max=None):
        """
        Iteratively search the counterfactuals based on gradient descent.
        """
        # Calculate the layer representation of baselines
        data = call_model_function(baselines, call_model_args=call_model_args, expected_keys=[REP_LAYER_VALUES])
        call_model_args = call_model_args or {}
        call_model_args.update({'layer_baseline': data[REP_LAYER_VALUES].detach()})

        # Iteratively search in sample space to close the reference rep
        delta = np.zeros_like(x_value)
        path = [x_value.copy()]

        for i in range(steps):
            data = call_model_function(x_value + delta, call_model_args=call_model_args, expected_keys=[REP_DISTANCE_GRADIENTS])
            grad = data[REP_DISTANCE_GRADIENTS]
            loss = np.linalg.norm(x_value + delta - baselines.mean(axis=0)) ** 2  # Approximate loss

            grad = normalize_by_2norm(grad)
            delta = delta + grad * step_size
            if clip_min_max:
                delta = np.clip(x_value + delta, clip_min_max[0], clip_min_max[1]) - x_value

            x_adv = x_value + delta
            if i % 100 == 0:
                print(f'{i} iterations, rep distance Loss {loss:.4f}')
            path.append(x_adv.copy())

        return np.array(path)

    def GetMask(self, **kwargs):
        # Convert x_value and baseline_features to NumPy
        x_value = kwargs["x_value"].detach().cpu().numpy() if isinstance(kwargs["x_value"], torch.Tensor) else kwargs["x_value"]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs["baseline_features"].detach().cpu().numpy() if isinstance(kwargs["baseline_features"], torch.Tensor) else kwargs["baseline_features"]
        x_steps = kwargs.get("x_steps", 201)  # Match IG2 default
        step_size = kwargs.get("step_size", 1.0)  # Adjustable step size
        clip_min_max = kwargs.get("clip_min_max", None)  # Disable clipping by default for feature space
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0)

        # Batch x_value to match baselines shape
        x_value = np.asarray([x_value], dtype=np.float32)
        baselines = np.asarray(baseline_features, dtype=np.float32)
        x_value = np.repeat(x_value, baselines.shape[0], axis=0)

        # GradPath search
        print('GradPath search...')
        path = self.Get_GradPath(x_value, baselines, call_model_function, call_model_args, x_steps, step_size, clip_min_max)
        np.testing.assert_allclose(x_value, path[0], rtol=0.01)

        # Integrate gradients on GradPath
        print('Integrate gradients on GradPath...')
        attr = np.zeros_like(x_value, dtype=np.float32)
        x_old = x_value

        for i, x_step in enumerate(path[1:], 1):
            x_old_tensor = torch.tensor(x_old, dtype=torch.float32, device=device)
            call_model_output = call_model_function(
                x_old_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=[INPUT_OUTPUT_GRADIENTS]
            )
            gradients = call_model_output[INPUT_OUTPUT_GRADIENTS]
            print(f"Step {i}, gradients shape: {gradients.shape}, norm: {np.linalg.norm(gradients):.4f}, min: {gradients.min():.4e}, max: {gradients.max():.4e}")

            # IG² attribution update
            attr += (x_old - x_step) * gradients
            x_old = x_step

        # Average over baselines
        result = np.mean(attr, axis=0)
        print(f"Raw attribution norm: {np.linalg.norm(result):.4f}, min: {result.min():.4e}, max: {result.max():.4e}")
        return result
