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

    def GetMask(self, x_value, call_model_function, model, call_model_args=None,
                x_baseline=None, x_steps=25, batch_size=1, device="cpu"):
        """Returns an integrated gradients mask for feature inputs.

        Args:
            x_value: Input tensor of features (torch.Tensor, shape: [N, D]).
            call_model_function: Function to interface with the CLAM model.
            model: CLAM model instance.
            call_model_args: Arguments for the model call (e.g., target class index).
            x_baseline: Baseline features (torch.Tensor, shape: [N, D]). Defaults to zeros.
            x_steps: Number of integration steps between baseline and input.
            batch_size: Number of steps to process in a batch.
            device: Device to perform computations on (e.g., 'cuda' or 'cpu').

        Returns:
            Integrated gradients mask (numpy array, shape: [N, D]).
        """
        # Set baseline to zeros if not provided
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
                call_model_output = call_model_function(
                    x_step_batched.reshape(-1, x_step_batched.shape[-1]),  # Flatten for model
                    model,
                    call_model_args=call_model_args,
                    expected_keys=self.expected_keys
                )

                self.format_and_check_call_model_output(
                    call_model_output,
                    x_step_batched.reshape(-1, x_step_batched.shape[-1]).shape,
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
