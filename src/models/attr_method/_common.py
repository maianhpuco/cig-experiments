import torch
import h5py 
import saliency.core as saliency 
import numpy as np 
from tqdm import tqdm 
import random 
'''
FOR CLAM MODEL: 
        # logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)  
'''
from saliency.core.base import CoreSaliency, INPUT_OUTPUT_GRADIENTS 


def call_model_function(features, model, call_model_args=None, expected_keys=None):
    device = next(model.parameters()).device
    features = features.to(device)
    features.requires_grad_(True)
    model.eval()

    was_batched = features.dim() == 3
    if was_batched:
        features = features.squeeze(0)  # [1, N, D] -> [N, D]
        
    model_output = model(features, [features.shape[0]])
    logits = model_output[0] if isinstance(model_output, tuple) else model_output

    target_class_idx = call_model_args['target_class_idx']
    target_logit = logits[:, target_class_idx]  # shape: [N] — no .sum() here!
    # print(f">>>>>>> Target logit shape: {target_logit.shape}")  # should be [N]
    grads = torch.autograd.grad(
        outputs=target_logit,
        inputs=features,
        grad_outputs=torch.ones_like(target_logit),
        create_graph=False,
        retain_graph=False
    )[0]

    gradients = grads.detach().cpu().numpy()
    if was_batched:
        gradients = np.expand_dims(gradients, axis=0)  # shape: [1, N, D] 
    # print(f">>>>>>> Gradients shape: {gradients.shape}")  # should be [N, D]
    return {INPUT_OUTPUT_GRADIENTS: gradients}
    

def get_mean_std_for_normal_dist(dataset):
    # Initialize accumulators
    feature_sum = None
    feature_sq_sum = None
    total_samples = 0

    for i in tqdm(range(len(dataset)), desc="Computing the Mean and Std"):
        sample = dataset[i]
        features = torch.tensor(sample['features'], dtype=torch.float32)  # Convert to tensor

        if feature_sum is None:
            feature_sum = torch.zeros_like(features.sum(dim=0))
            feature_sq_sum = torch.zeros_like(features.sum(dim=0))

        feature_sum += features.sum(dim=0)
        feature_sq_sum += (features ** 2).sum(dim=0)
        total_samples += features.shape[0]  # Number of patches

    # Compute mean and std
    mean = feature_sum / total_samples
    std = torch.sqrt((feature_sq_sum / total_samples) - (mean ** 2))
    
    return mean, std


def sample_random_features(dataset, num_files=10):
    basenames = dataset.basenames[:]
    random.shuffle(basenames)
    selected_basenames = random.sample(basenames, min(num_files, len(dataset)))
    stacked_features = []
    with tqdm(total=len(selected_basenames), desc="Loading Feature Files") as pbar:
        for basename in selected_basenames:
            sample = dataset[dataset.basenames.index(basename)]  # Get dataset sample

            features = torch.tensor(sample['features'], dtype=torch.float32)  # Convert to tensor
            stacked_features.append(features)

            pbar.update(1)

    # Stack all features together along the first dimension
    stacked_features = torch.cat(stacked_features, dim=0)
    return stacked_features, selected_basenames
