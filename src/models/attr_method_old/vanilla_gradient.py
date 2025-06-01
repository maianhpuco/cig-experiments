import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
from attr_method._common import PreprocessInputs, call_model_function 

class VanillaGradients(CoreSaliency):
    """Vanilla Gradient Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value").to(kwargs.get("device", "cpu"))
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)

        x_value_tensor = x_value.clone().detach().requires_grad_(True).to(x_value.device)

        call_model_output = call_model_function(
            x_value_tensor,
            model,
            call_model_args=call_model_args,
            expected_keys=self.expected_keys
        )

        self.format_and_check_call_model_output(
            call_model_output,
            x_value_tensor.shape,
            self.expected_keys
        )

        gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS]  # Expected: np.ndarray
        gradients = torch.tensor(gradients_batch, device=x_value.device).squeeze(0)  # [N, D]

        return gradients.detach().cpu().numpy()  # Return numpy for consistency with others

