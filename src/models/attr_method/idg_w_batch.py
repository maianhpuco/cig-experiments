import os
import numpy as np
import torch
from tqdm import tqdm
import saliency.core as saliency
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS
from torch.amp import autocast  # Updated import


class IDG(CoreSaliency):
    """Integrated Decision Gradients with slope-weighted redistribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    @staticmethod
    def getSlopes(x_baseline_batch, x_value, model, x_steps, device, call_model_function, call_model_args=None, target_class_idx=0): 
        alphas = torch.linspace(0, 1, x_steps, device=device)
        logits = torch.zeros(x_steps, device=device)
        slopes = torch.zeros(x_steps, device=device)
        x_diff = x_value - x_baseline_batch

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing Slopes", ncols=100)):
            with torch.no_grad():
                x_step_batch = (x_baseline_batch + alpha * x_diff).to(device)
                with autocast('cuda'):  # Updated autocast
                    model_output = call_model_function(
                        x_step_batch,
                        model,
                        call_model_args=call_model_args,
                        expected_keys=None
                    )
                logits_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
                logit = logits_tensor[0, target_class_idx] if logits_tensor.dim() == 2 else logits_tensor[target_class_idx]
                logits[step_idx] = logit
                del x_step_batch, model_output, logits_tensor
                torch.cuda.empty_cache()

        delta_alpha = float(alphas[1] - alphas[0])
        slopes[1:] = (logits[1:] - logits[:-1]) / (delta_alpha + 1e-9)
        slopes[0] = 0.0
        return slopes, delta_alpha, logits

    @staticmethod
    def getAlphaParameters(slopes, steps, step_size):
        normed = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + 1e-8)
        normed[0] = 0
        weights = normed / (torch.sum(normed) + 1e-8)

        placements_float = weights * steps
        placements_int = placements_float.floor().int()
        remaining = steps - torch.sum(placements_int)

        float_copy = placements_float.clone()
        float_copy[torch.where(placements_int != 0)] = -1
        top_idx = torch.flip(torch.sort(float_copy)[1], dims=[0])
        placements_int[top_idx[:remaining]] += 1

        alphas = torch.zeros(steps, device=slopes.device)
        alpha_steps = torch.zeros(steps, device=slopes.device)
        start_idx = 0
        start_val = 0

        for count in placements_int:
            if count == 0:
                continue
            count = int(count.item())
            new_alphas = torch.linspace(start_val, start_val + step_size, count + 1, device=slopes.device)[:count]
            alphas[start_idx:start_idx + count] = new_alphas
            alpha_steps[start_idx:start_idx + count] = (step_size / count)
            start_idx += count
            start_val += step_size

        return alphas, alpha_steps

    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")  # [1, N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features")  # [1, N, D]
        x_steps = kwargs.get("x_steps", 50)  # Match your current setting
        device = kwargs.get("device", "cpu")
        target_class_idx = call_model_args.get("target_class_idx", 0) if call_model_args else 0
        batch_size = kwargs.get("batch_size", 500)  # Match your current setting
        memmap_path = kwargs.get("memmap_path", "./temp_idg")  # Path for temporary files

        # Create temporary directory for intermediate results
        os.makedirs(memmap_path, exist_ok=True)

        # Device and format checks
        x_value = x_value.to(device, dtype=torch.float32)
        baseline_features = baseline_features.to(device, dtype=torch.float32)

        # Sample baseline with shape [1, N, D]
        sample_idx = torch.randint(0, baseline_features.size(0), (x_value.size(0),), device=device)
        x_baseline_batch = baseline_features[sample_idx]  # [1, N, D]
        x_diff = x_value - x_baseline_batch  # [1, N, D]

        # Compute slopes
        slopes, _, _ = self.getSlopes(
            x_baseline_batch, x_value, model, x_steps, device, call_model_function, call_model_args, target_class_idx
        )
        alphas, alpha_sizes = self.getAlphaParameters(slopes, x_steps, 1.0 / x_steps)

        num_instances = x_value.size(1)  # N
        batch_size = min(batch_size, num_instances)
        temp_files = []

        # Process instances in batches
        for batch_start in range(0, num_instances, batch_size):
            batch_end = min(batch_start + batch_size, num_instances)
            x_value_batch = x_value[:, batch_start:batch_end, :]  # [1, batch_size, D]
            x_baseline_batch_batch = x_baseline_batch[:, batch_start:batch_end, :]  # [1, batch_size, D]
            x_diff_batch = x_diff[:, batch_start:batch_end, :]  # [1, batch_size, D]
            integrated_batch = torch.zeros_like(x_value_batch, dtype=torch.float32, device=device)  # [1, batch_size, D]
            prev_logit = None
            slope_cache = torch.zeros(x_steps, device=device)

            # for step_idx, (alpha, step_size) in enumerate(tqdm(zip(alphas, alpha_sizes), total=x_steps, desc=f"Computing IDG (batch {batch_start}-{batch_end})", ncols=100)):                x_step = (x_baseline_batch_batch + alpha * x_diff_batch).detach().requires_grad_(True)  # [1, batch_size, D]
            for step_idx, alpha, step_size in enumerate(zip(alphas, alpha_sizes)):
                with autocast('cuda'):  # Updated autocast
                    # Compute gradients
                    call_output = call_model_function(
                        x_step, model, call_model_args=call_model_args, expected_keys=self.expected_keys
                    )
                    # Compute logits
                    model_output = call_model_function(
                        x_step, model, call_model_args=call_model_args, expected_keys=None
                    )
                logits_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
                logit = logits_tensor[0, target_class_idx] if logits_tensor.dim() == 2 else logits_tensor[target_class_idx]

                grads = call_output[INPUT_OUTPUT_GRADIENTS]  # [1, batch_size, D]
                grads_avg = torch.tensor(grads, dtype=torch.float32, device=device)  # [1, batch_size, D]

                if prev_logit is not None:
                    alpha_diff = alpha - alphas[step_idx - 1]
                    slope_cache[step_idx] = (logit - prev_logit) / (alpha_diff + 1e-9)
                prev_logit = logit

                integrated_batch += grads_avg * slope_cache[step_idx] * step_size

                del x_step, grads_avg, call_output, model_output, logits_tensor
                torch.cuda.empty_cache()

            # Save integrated_batch to disk
            temp_file = os.path.join(memmap_path, f"integrated_batch_{batch_start}_{batch_end}.npy")
            np.save(temp_file, integrated_batch.detach().cpu().numpy())  # Added .detach()
            temp_files.append(temp_file)
            del x_value_batch, x_baseline_batch_batch, x_diff_batch, integrated_batch
            torch.cuda.empty_cache()

        # Read back and aggregate integrated results
        integrated = torch.zeros_like(x_value, dtype=torch.float32, device='cpu')  # [1, N, D]
        for temp_file in temp_files:
            batch_data = np.load(temp_file)
            batch_start = int(temp_file.split('_')[-2])
            batch_end = int(temp_file.split('_')[-1].replace('.npy', ''))
            integrated[:, batch_start:batch_end, :] = torch.from_numpy(batch_data)
            os.remove(temp_file)  # Clean up temporary file
        temp_files.clear()

        # Compute final attribution on CPU
        attribution = integrated * x_diff.to('cpu')  # [1, N, D]
        return attribution.numpy()