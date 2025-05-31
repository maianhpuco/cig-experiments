import numpy as np
import torch
import saliency.core as saliency
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from tqdm import tqdm

EPSILON = 1e-9

def call_model_function(features, model, call_model_args=None, expected_keys=None):
    """Compute model logits and gradients for saliency with CLAM model."""
    device = next(model.parameters()).device  # Get model's device
    features = features.to(device)  # Ensure features are on the same device
    features.requires_grad_(True)  # Enable gradient computation
    model.eval()

    # CLAM model expects features and batch size
    model_output = model(features, [features.shape[0]])

    # Handle CLAM's tuple output (logits, probs, pred, etc.)
    if isinstance(model_output, tuple):
        logits = model_output[0]  # Assume first element is logits
    else:
        logits = model_output

    # Get target class index from call_model_args
    class_idx_str = 'class_idx_str'
    target_class_idx = call_model_args[class_idx_str] 
    target_logit = logits[:, target_class_idx].sum()  # Sum logits for the target class

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=target_logit,
        inputs=features,
        grad_outputs=torch.ones_like(target_logit),
        create_graph=False,
        retain_graph=False
    )[0]

    gradients = grads.detach().cpu().numpy()  # Convert to numpy for saliency
    return {INPUT_OUTPUT_GRADIENTS: gradients}

class IntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""
    expected_keys = [INPUT_OUTPUT_GRADIENTS]
    
    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # torch.Tensor, shape [N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)  # torch.Tensor, shape [N, D]
        x_steps = kwargs.get("x_steps", 25)
        batch_size = kwargs.get("batch_size", 1)  # Add batch_size parameter for efficiency
        device = kwargs.get("device", "cpu")

        # Default to zeros baseline if none provided
        if baseline_features is None:
            baseline_features = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        
        # Ensure baseline is on the correct device and has the same shape as x_value
        baseline_features = baseline_features.to(device)
        assert baseline_features.shape == x_value.shape, \
            f"Baseline shape {baseline_features.shape} must match input shape {x_value.shape}"

        # Allocate result tensor
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)
        x_diff = x_value - baseline_features  # Shape [N, D]

        # Generate interpolation steps
        alphas = np.linspace(0, 1, x_steps)
        x_step_batched = []
        for alpha in tqdm(alphas, desc="Computing Integrated Gradients:", ncols=100):
            x_step = baseline_features + alpha * x_diff  # Shape [N, D]
            x_step_batched.append(x_step)
            if len(x_step_batched) == batch_size or alpha == alphas[-1]:
                x_step_batched = torch.stack(x_step_batched).to(device)  # Shape [batch_size, N, D]
                flat_batch = x_step_batched.reshape(-1, x_value.shape[-1])  # Shape [batch_size * N, D]
                flat_batch.requires_grad_(True)

                call_model_output = call_model_function(
                    flat_batch,
                    model,
                    call_model_args=call_model_args,
                    expected_keys=self.expected_keys
                )

                self.format_and_check_call_model_output(
                    call_model_output, flat_batch.shape, self.expected_keys
                )

                gradients = torch.tensor(
                    call_model_output[INPUT_OUTPUT_GRADIENTS], device=device
                ).reshape(len(x_step_batched), x_value.shape[0], x_value.shape[-1])  # Shape [batch_size, N, D]
                attribution_values += gradients.sum(dim=0)  # Sum over batch, shape [N, D]
                x_step_batched = []

        # Compute final attribution
        attribution_values = attribution_values * x_diff / x_steps
        return attribution_values.cpu().numpy()