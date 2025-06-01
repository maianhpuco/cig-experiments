class PICTestFunctions:
    @staticmethod
    def generate_random_mask(num_patches, fraction=0.01):
        """Generate a random binary mask for PIC metrics."""
        num_ones = int(num_patches * fraction)
        mask = np.zeros(num_patches, dtype=np.int64)
        indices = np.random.choice(num_patches, num_ones, replace=False)
        mask[indices] = 1
        return mask

    @staticmethod
    def compute_pic_metric(features, saliency_map, random_mask, thresholds, mode, model, device):
        """Compute PIC metric (SIC or AIC) for feature-based saliency."""
        class MetricResult:
            def __init__(self, auc):
                self.auc = auc

        # Sort patches by saliency
        sorted_indices = np.argsort(saliency_map)[::-1]
        num_patches = len(saliency_map)
        steps = [int(t * num_patches) for t in thresholds]
        scores = np.zeros(len(thresholds))

        features = torch.tensor(features, dtype=torch.float32).to(device)
        features = features.unsqueeze(0)  # Shape: [1, N, 1024]
        with torch.no_grad():
            output = model(features, [features.shape[1]])
            logits = output[0]
            probs = torch.softmax(logits, dim=1)
            target_class = torch.argmax(probs, dim=1).item()
            base_score = probs[0, target_class].item()

        for i, step in enumerate(steps):
            masked_features = features.clone()
            if mode == 0:  # SIC: keep top patches
                indices = sorted_indices[:step]
            else:  # AIC: remove top patches
                indices = sorted_indices[step:]
            masked_features[0, indices] = 0  # Zero out patches
            with torch.no_grad():
                output = model(masked_features, [masked_features.shape[1]])
                logits = output[0]
                probs = torch.softmax(logits, dim=1)
                scores[i] = probs[0, target_class].item()

        # Compute AUC
        auc_score = (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (len(scores) - 1)
        return MetricResult(auc=auc_score)

# Provided CausalMetric and auc
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric:
    def __init__(self, model, num_patches: int, mode: str, step: int, substrate_fn):
        assert mode in ['del', 'ins']
        self.model = model
        self.num_patches = num_patches
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, feature_tensor: torch.Tensor, saliency_map: np.ndarray, device: str, max_batch_size: int = 50):
        n_steps = (self.num_patches + self.step - 1) // self.step
        batch_size = min(n_steps, max_batch_size)

        if batch_size > n_steps:
            print(f"Batch size cannot be greater than number of steps: {n_steps}")
            return 0, []

        original_pred = self.model(feature_tensor.to(device))
        _, index = torch.max(original_pred[0], 1)
        target_class = index.item()
        percentage = torch.nn.functional.softmax(original_pred[0], dim=1)
        original_pred = percentage[target_class].item()

        scores = np.zeros(n_steps + 1)

        if self.mode == 'del':
            start = feature_tensor.clone()
            finish = self.substrate_fn(feature_tensor)
            black_pred = self.model(finish.to(device))
            percentage = torch.nn.functional.softmax(black_pred[0], dim=1)
            black_pred = percentage[target_class].item()
            scores[0] = original_pred
        elif self.mode == 'ins':
            start = self.substrate_fn(feature_tensor)
            finish = feature_tensor.clone()
            neutral_pred = self.model(start.to(device))
            percentage = torch.nn.functional.softmax(neutral_pred[0], dim=1)
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
            percentage = torch.nn.functional.softmax(output[0], dim=1)
            scores[total_steps - batch:total_steps] = percentage[:, target_class].cpu().numpy()

        return n_steps, scores