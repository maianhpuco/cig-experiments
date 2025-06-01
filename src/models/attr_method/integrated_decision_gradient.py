import numpy as np
import torch
import saliency.core as saliency
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from tqdm import tqdm

EPSILON = 1e-9


def getGradientsParallel(inputs, model, target_class):
    """Compute gradients and logits for a batch of feature inputs."""
    inputs = inputs.to(next(model.parameters()).device)  # Ensure inputs are on model device
    inputs.requires_grad_(True)
    model.eval()

    # CLAM model returns (logits, probs, pred)
    output = model(inputs, [inputs.shape[0]])
    logits, _, _ = output  # Unpack logits, ignore probs and pred

    scores = logits[:, target_class].sum()  # Sum logits for the target class
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=inputs,
        grad_outputs=torch.ones_like(scores),
        create_graph=False,
        retain_graph=False
    )[0]

    return gradients.detach(), scores.detach()
 

def getPredictionParallel(inputs, model, target_class):
    """Compute logit scores for a batch of feature inputs."""
    inputs = inputs.to(next(model.parameters()).device)
    model.eval()

    # CLAM model returns (logits, probs, pred)
    output = model(inputs, [inputs.shape[0]])
    logits, _, _ = output  # Unpack logits, ignore probs and pred

    scores = logits[:, target_class]
    return scores.detach() 
    
def getSlopes(baseline, baseline_diff, model, steps, batch_size, device, target_class):
    """Compute logit slopes for IDG."""
    if steps % batch_size != 0:
        print(f"Steps ({steps}) must be evenly divisible by batch size ({batch_size})!")
        return 0, 0

    loops = int(steps / batch_size)

    # Generate alpha values
    alphas = torch.linspace(0, 1, steps).to(device)  # Shape: [steps]
    logits = torch.zeros(steps).to(device)  # Store logits

    # Run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_features = baseline + alphas[start:end].reshape(-1, 1, 1) * baseline_diff
        interp_features = interp_features.reshape(-1, baseline_diff.shape[-1])  # Flatten for model

        logits[start:end] = getPredictionParallel(interp_features, model, target_class)

    # Calculate slopes
    slopes = torch.zeros(steps).to(device)
    x_diff = float(alphas[1] - alphas[0])  # Step size

    slopes[0] = 0
    for i in range(steps - 1):
        y_diff = logits[i + 1] - logits[i]
        slopes[i + 1] = y_diff / x_diff

    return slopes, x_diff

def getAlphaParameters(slopes, steps, step_size):
    """Compute non-uniform alpha values based on slopes."""
    # Normalize slopes to [0, 1]
    slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + EPSILON)
    slopes_0_1_norm[0] = 0  # Reset first slope to zero
    slopes_sum_1_norm = slopes_0_1_norm / (torch.sum(slopes_0_1_norm) + EPSILON)  # Normalize to sum to 1

    # Distribute steps based on slopes
    sample_placements_float = slopes_sum_1_norm * steps
    sample_placements_int = sample_placements_float.type(torch.int)
    remaining_to_fill = steps - torch.sum(sample_placements_int)

    # Handle remaining steps
    sample_placements_float[torch.where(sample_placements_int != 0)[0]] = -1
    remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims=[0])
    sample_placements_int[remaining_hi_lo[:remaining_to_fill]] = 1

    # Generate new alpha values
    alphas = torch.zeros(steps)
    alpha_substep_size = torch.zeros(steps)

    alpha_start_index = 0
    alpha_start_value = 0

    for num_samples in sample_placements_int:
        if num_samples == 0:
            continue
        alphas[alpha_start_index:alpha_start_index + num_samples] = torch.linspace(
            alpha_start_value, alpha_start_value + step_size, num_samples + 1
        )[:num_samples]
        alpha_substep_size[alpha_start_index:alpha_start_index + num_samples] = step_size / num_samples
        alpha_start_index += num_samples
        alpha_start_value += step_size

    return alphas, alpha_substep_size 

def IDG(input, model, steps, batch_size, baseline, device, target_class):
    """Compute Integrated Decision Gradient for feature inputs with CLAM model."""
    if steps % batch_size != 0:
        print(f"Steps ({steps}) must be evenly divisible by batch size ({batch_size})!")
        return 0

    loops = int(steps / batch_size)

    # Handle baseline
    if not torch.is_tensor(baseline):
        baseline = torch.full_like(input, baseline, dtype=torch.float32)
    baseline = baseline.to(device)
    input = input.to(device)
    baseline_diff = input - baseline

    # Compute slopes
    slopes, step_size = getSlopes(baseline, baseline_diff, model, steps, batch_size, device, target_class)
    alphas, alpha_substep_size = getAlphaParameters(slopes, steps, step_size)

    alphas = alphas.to(device).reshape(steps, 1, 1)
    alpha_substep_size = alpha_substep_size.to(device).reshape(steps, 1, 1)

    # Arrays to store gradients and logits
    gradients = torch.zeros((steps, input.shape[1])).to(device)
    logits = torch.zeros(steps).to(device)

    # Run batched input
    for i in tqdm(range(loops), desc="Computing gradients"):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_features = baseline + alphas[start:end] * baseline_diff
        interp_features = interp_features.reshape(-1, input.shape[-1])

        gradients[start:end], logits[start:end] = getGradientsParallel(interp_features, model, target_class)

    # Calculate slopes
    slopes = torch.zeros(steps).to(device)
    slopes[0] = 0
    for i in range(steps - 1):
        slopes[i + 1] = (logits[i + 1] - logits[i]) / (alphas[i + 1] - alphas[i])

    # Weight gradients by slopes and alpha substep sizes
    gradients = gradients * slopes.reshape(steps, 1) * alpha_substep_size.reshape(steps, 1)

    # Compute integral approximation
    grads = gradients.mean(dim=0)
    grads = grads * baseline_diff[0]

    return grads 