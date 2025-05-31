import torch
import numpy as np
'''
Input Shape: Handles feature tensors [1, N, 512] instead of image tensors [1, 3, H, W].
Baseline: Uses substrate_fn to apply the mean feature vector for insertion and zero vector for deletion.
Saliency Map: Processes 1D saliency scores [N,] instead of 2D maps [H, W].
Steps: Based on num_patches instead of img_hw * img_hw. 
'''
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric:
    def __init__(self, model, num_patches: int, mode: str, step: int, substrate_fn: Callable):
        """
        Create deletion/insertion metric instance for features.
        
        Args:
            model: Black-box model being explained.
            num_patches: Number of patches in the WSI.
            mode: 'del' or 'ins'.
            step: Number of patches modified per iteration.
            substrate_fn: Function mapping original features to baseline features.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.num_patches = num_patches
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, feature_tensor: torch.Tensor, saliency_map: np.ndarray, device: str, max_batch_size: int = 50):
        """
        Run metric on one feature-saliency pair.
        
        Args:
            feature_tensor: Feature tensor of shape [1, N, 512].
            saliency_map: Saliency scores of shape [N,].
            device: GPU or CPU.
            max_batch_size: Maximum batch size for processing.
        
        Returns:
            n_steps: Number of steps used.
            scores: Array of model confidence scores at each step.
        """
        n_steps = (self.num_patches + self.step - 1) // self.step
        batch_size = min(n_steps, max_batch_size)

        if batch_size > n_steps:
            print(f"Batch size cannot be greater than number of steps: {n_steps}")
            return 0, []

        original_pred = self.model(feature_tensor.to(device))
        _, index = torch.max(original_pred, 1)
        target_class = index[0]
        percentage = torch.nn.functional.softmax(original_pred, dim=1)[0]
        original_pred = percentage[target_class].item()

        scores = np.zeros(n_steps + 1)

        if self.mode == 'del':
            start = feature_tensor.clone()
            finish = self.substrate_fn(feature_tensor)
            black_pred = self.model(finish.to(device))
            percentage = torch.nn.functional.softmax(black_pred, dim=1)[0]
            black_pred = percentage[target_class].item()
            scores[0] = original_pred
        elif self.mode == 'ins':
            start = self.substrate_fn(feature_tensor)
            finish = feature_tensor.clone()
            neutral_pred = self.model(start.to(device))
            percentage = torch.nn.functional.softmax(neutral_pred, dim=1)[0]
            neutral_pred = percentage[target_class].item()
            scores[0] = neutral_pred

        salient_order = np.flip(np.argsort(saliency_map, axis=0), axis=0)

        density = np.zeros(n_steps + 1)
        if self.mode == "del":
            density[0] = 1
        elif self.mode == "ins":
            density[0] = 0

        total_steps = 1
        num_batches = n_steps // batch_size
        leftover = n_steps % batch_size
        batches = np.full(num_batches + 1, batch_size)
        batches[-1] = leftover

        for batch in batches:
            if batch == 0:
                continue
            features_batch = torch.zeros((batch, feature_tensor.shape[1], feature_tensor.shape[2]))

            for i in range(batch):
                coords = salient_order[self.step * (total_steps - 1):self.step * total_steps]
                start[0, coords] = finish[0, coords]
                features_batch[i] = start
                total_steps += 1

            output = self.model(features_batch.to(device)).detach()
            percentage = torch.nn.functional.softmax(output, dim=1)
            scores[total_steps - batch:total_steps] = percentage[:, target_class].cpu().numpy()

        return n_steps, scores