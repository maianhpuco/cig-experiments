import torch
import numpy as np
from typing import Callable

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class ModelWrapper:
    """Wraps a model to standardize forward calls for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model, handling both single and batched inputs.
        
        Args:
            input: Tensor of shape [N, D] or [B, N, D] where B is batch size, N is number of patches, D is feature dimension.
        
        Returns:
            logits: Tensor of shape [B, C] where C is number of classes, or [C] for single input.
        """
        if self.model_type != 'clam':
            return self.model(input)

        # Handle batched input [B, N, D]
        if input.dim() == 3:
            batch_size = input.shape[0]
            logits_list = []
            for i in range(batch_size):
                single_input = input[i]  # Shape [N, D]
                instance_per_slide = [single_input.shape[0]]
                output = self.model(single_input, instance_per_slide)
                logits = output[0] if isinstance(output, tuple) else output
                logits_list.append(logits)
            return torch.stack(logits_list)  # Shape [B, C]
        
        # Handle single input [N, D]
        if input.dim() == 2:
            instance_per_slide = [input.shape[0]]
            output = self.model(input, instance_per_slide)
            logits = output[0] if isinstance(output, tuple) else output
            return logits
        
        raise ValueError(f"Unsupported input shape: {input.shape}")

class CausalMetric:
    def __init__(self, model, num_patches: int, mode: str, step: int, substrate_fn: Callable):
        """
        Create deletion/insertion metric instance for features.
        
        Args:
            model: Black-box model being explained (CLAM model).
            num_patches: Number of patches in the WSI feature tensor.
            mode: 'del' or 'ins' (deletion or insertion).
            step: Number of patches modified per iteration.
            substrate_fn: Function mapping original features to baseline features.
        """
        assert mode in ['del', 'ins']
        self.model = ModelWrapper(model, model_type='clam')  # Wrap the model
        self.num_patches = num_patches
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, feature_tensor: torch.Tensor, saliency_map: np.ndarray, device: str, max_batch_size: int = 50):
        """
        Run RISE metric on one feature-saliency pair.
        
        Args:
            feature_tensor: Feature tensor of shape [1, N, 512].
            saliency_map: Saliency scores of shape [N,].
            device: 'cuda' or 'cpu'.
            max_batch_size: Maximum batch size for processing.
        
        Returns:
            tuple: (n_steps, scores)
                - n_steps: Number of steps used.
                - scores: Array of model confidence scores at each step.
        """
        n_steps = (self.num_patches + self.step - 1) // self.step
        batch_size = min(n_steps, max_batch_size)

        if batch_size > n_steps:
            print(f"Batch size cannot be greater than number of steps: {n_steps}")
            return 0, []

        # Get original prediction using ModelWrapper
        self.model.model.eval()  # Ensure the underlying model is in eval mode
        with torch.no_grad():
            original_pred = self.model.forward(feature_tensor.to(device))
            _, index = torch.max(original_pred, 1)
            target_class = index[0]
            percentage = torch.nn.functional.softmax(original_pred, dim=1)[0]
            original_pred_score = percentage[target_class].item()

        scores = np.zeros(n_steps + 1)

        # Initialize start and finish tensors
        if self.mode == 'del':
            start = feature_tensor.clone()
            finish = self.substrate_fn(feature_tensor)
            scores[0] = original_pred_score
        elif self.mode == 'ins':
            start = self.substrate_fn(featurekeywordflow

System: It looks like the provided code and error trace indicate an issue with the input tensor shape in the `ModelWrapper.forward` method when handling batched inputs for the CLAM model. The error `RuntimeError: self must be a matrix` suggests that the CLAM model expects a 2D tensor `[N, D]` (number of patches, feature dimension), but the input tensor is likely not properly formatted when passed to the model.

The `CausalMetric.single_run` method in `RISETestFunctions.py` creates a `features_batch` tensor with shape `[batch, N, 512]`, which is passed to `ModelWrapper.forward`. The current `forward` method attempts to handle 3D inputs by squeezing the first dimension, but this is insufficient for batched inputs, as the CLAM model requires a 2D tensor for each sample, and the batch dimension needs explicit handling.

The updated code you provided in the latest message is incomplete, as it cuts off in the `single_run` method. However, based on the error and the previous context, I'll provide a corrected version of `RISETestFunctions.py` that:
1. Fixes the `ModelWrapper.forward` method to properly handle both single inputs `[N, D]` and batched inputs `[B, N, D]` by processing each sample in the batch individually.
2. Retains the fix for negative strides (using `.copy()` on `salient_order`).
3. Ensures compatibility with the CLAM model and the `metric_rise_clam.py` script.

Here’s the corrected code for `RISETestFunctions.py`:

<xaiArtifact artifact_id="c65c554d-d4a2-452a-86cc-72b71369eb8a" artifact_version_id="47feb966-4164-4183-a61e-769f3853b8b2" title="RISETestFunctions.py" contentType="text/python">
import torch
import numpy as np
from typing import Callable

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class ModelWrapper:
    """Wraps a model to standardize forward calls for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model, handling both single and batched inputs.
        
        Args:
            input: Tensor of shape [N, D] or [B, N, D] where B is batch size, N is number of patches, D is feature dimension.
        
        Returns:
            logits: Tensor of shape [B, C] where C is number of classes, or [C] for single input.
        """
        if self.model_type != 'clam':
            return self.model(input)

        # Handle batched input [B, N, D]
        if input.dim() == 3:
            batch_size = input.shape[0]
            logits_list = []
            for i in range(batch_size):
                single_input = input[i]  # Shape [N, D]
                instance_per_slide = [single_input.shape[0]]
                output = self.model(single_input, instance_per_slide)
                logits = output[0] if isinstance(output, tuple) else output
                logits_list.append(logits)
            return torch.stack(logits_list)  # Shape [B, C]
        
        # Handle single input [N, D]
        if input.dim() == 2:
            instance_per_slide = [input.shape[0]]
            output = self.model(input, instance_per_slide)
            logits = output[0] if isinstance(output, tuple) else output
            return logits
        
        raise ValueError(f"Unsupported input shape: {input.shape}")

class CausalMetric:
    def __init__(self, model, num_patches: int, mode: str, step: int, substrate_fn: Callable):
        """
        Create deletion/insertion metric instance for features.
        
        Args:
            model: Black-box model being explained (CLAM model).
            num_patches: Number of patches in the WSI feature tensor.
            mode: 'del' or 'ins' (deletion or insertion).
            step: Number of patches modified per iteration.
            substrate_fn: Function mapping original features to baseline features.
        """
        assert mode in ['del', 'ins']
        self.model = ModelWrapper(model, model_type='clam')  # Wrap the model
        self.num_patches = num_patches
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, feature_tensor: torch.Tensor, saliency_map: np.ndarray, device: str, max_batch_size: int = 50):
        """
        Run RISE metric on one feature-saliency pair.
        
        Args:
            feature_tensor: Feature tensor of shape [1, N, 512].
            saliency_map: Saliency scores of shape [N,].
            device: 'cuda' or 'cpu'.
            max_batch_size: Maximum batch size for processing.
        
        Returns:
            tuple: (n_steps, scores)
                - n_steps: Number of steps used.
                - scores: Array of model confidence scores at each step.
        """
        n_steps = (self.num_patches + self.step - 1) // self.step
        batch_size = min(n_steps, max_batch_size)

        if batch_size > n_steps:
            print(f"Batch size cannot be greater than number of steps: {n_steps}")
            return 0, []

        # Get original prediction using ModelWrapper
        self.model.model.eval()  # Ensure the underlying model is in eval mode
        with torch.no_grad():
            original_pred = self.model.forward(feature_tensor.to(device))
            _, index = torch.max(original_pred, 1)
            target_class = index[0]
            percentage = torch.nn.functional.softmax(original_pred, dim=1)[0]
            original_pred_score = percentage[target_class].item()

        scores = np.zeros(n_steps + 1)

        # Initialize start and finish tensors
        if self.mode == 'del':
            start = feature_tensor.clone()
            finish = self.substrate_fn(feature_tensor)
            scores[0] = original_pred_score
        elif self.mode == 'ins':
            start = self.substrate_fn(feature_tensor)
            finish = feature_tensor.clone()
            with torch.no_grad():
                neutral_pred = self.model.forward(start.to(device))
                percentage = torch.nn.functional.softmax(neutral_pred, dim=1)[0]
                scores[0] = percentage[target_class].item()

        # Sort patches by saliency, ensuring positive strides
        salient_order = np.flip(np.argsort(saliency_map, axis=0), axis=0).copy()

        total_steps = 1
        num_batches = n_steps // batch_size
        leftover = n_steps % batch_size
        batches = np.full(num_batches + 1, batch_size)
        batches[-1] = leftover

        for batch in batches:
            if batch == 0:
                continue
            features_batch = torch.zeros((batch, feature_tensor.shape[1], feature_tensor.shape[2]), device=device)

            for i in range(batch):
                coords = salient_order[self.step * (total_steps - 1):self.step * total_steps]
                start[0, coords] = finish[0, coords]
                features_batch[i] = start
                total_steps += 1

            with torch.no_grad():
                output = self.model.forward(features_batch.to(device))
                percentage = torch.nn.functional.softmax(output, dim=1)
                scores[total_steps - batch:total_steps] = percentage[:, target_class].cpu().numpy()

        return n_steps, scores