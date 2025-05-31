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
        x_value = kwargs.get("x_value")  # torch.Tensor
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)  # torch.Tensor
        x_steps = kwargs.get("x_steps", 25)
        device = kwargs.get("device", "cpu")  

        # Allocate result tensor on device
        attribution_values = torch.zeros_like(x_value, dtype=torch.float32, device=device)

        alphas = np.linspace(0, 1, x_steps)
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]  # [1, N, D]
        x_diff = x_value - x_baseline_batch.squeeze(0)

        for alpha in tqdm(alphas, desc="Computing:", ncols=100):
            x_step_batch = x_baseline_batch + alpha * x_diff.unsqueeze(0)
            x_step_batch_tensor = x_step_batch.squeeze(0).clone().detach().requires_grad_(True).to(device)

            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )

            self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)

            gradients_batch_np = call_model_output[INPUT_OUTPUT_GRADIENTS]  # shape: (N, D), np.ndarray
            gradients_avg = torch.tensor(gradients_batch_np, device=device)  # Convert to tensor for torch ops

            attribution_values += gradients_avg

        x_diff = x_diff.reshape(-1, x_value.shape[-1]).to(device)  # (N, D)
        attribution_values = attribution_values * x_diff

        return attribution_values.cpu().numpy() / x_steps
 