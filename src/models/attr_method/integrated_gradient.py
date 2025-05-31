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
    """Implements Integrated Gradients for feature inputs with CLAM model."""
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs):
        """Returns an integrated gradients mask for feature inputs.

        Expected keys in kwargs:
            - x_value: Input features [N, D]
            - call_model_function: Function to get logits and gradients
            - model: CLAM model instance
            - call_model_args: dict with 'target_class_idx'
            - baseline_features: Baseline tensor [N, D] (optional)
            - x_steps: Number of integration steps
            - batch_size: Batch size for interpolation steps
            - device: Computation device
        """
        x_value = kwargs["x_value"]
        call_model_function = kwargs["call_model_function"]
        model = kwargs["model"]
        call_model_args = kwargs.get("call_model_args", None)
        device = kwargs.get("device", "cpu")
        x_steps = kwargs.get("x_steps", 25)
        batch_size = kwargs.get("batch_size", 1)

        # Allow use of 'baseline_features' as alias for 'x_baseline'
        x_baseline = kwargs.get("baseline_features", None)
        if x_baseline is None:
            x_baseline = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        assert x_baseline.shape == x_value.shape, "Baseline and input shapes must match"

        x_diff = x_value - x_baseline  # Difference between input and baseline
        total_gradients = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        # Generate interpolation steps
        x_step_batched = []
        for idx, alpha in enumerate(tqdm(np.linspace(0, 1, x_steps), desc="Integrated Gradients")):
            x_step = x_baseline + alpha * x_diff
            x_step_batched.append(x_step)
            if len(x_step_batched) == batch_size or alpha == 1:
                x_step_batched = torch.stack(x_step_batched).to(device)  # Shape: [batch_size, N, D]
                flat_batch = x_step_batched.reshape(-1, x_step_batched.shape[-1])
                
                call_model_output = call_model_function(
                    flat_batch,
                    model,
                    call_model_args=call_model_args,
                    expected_keys=self.expected_keys
                )

                self.format_and_check_call_model_output(
                    call_model_output,
                    flat_batch.shape,
                    self.expected_keys
                )

                # Reshape gradients to match input and sum
                gradients = torch.tensor(
                    call_model_output[INPUT_OUTPUT_GRADIENTS],
                    device=device
                ).reshape(len(x_step_batched), x_value.shape[0], x_value.shape[-1])
                total_gradients += gradients.sum(dim=0)  # Sum across batch
                x_step_batched = []

        # Compute final attribution
        attribution = total_gradients * x_diff / x_steps
        return attribution.cpu().numpy()
