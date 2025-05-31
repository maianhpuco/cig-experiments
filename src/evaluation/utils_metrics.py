import torch
from typing import Tuple

def compute_feature_bokeh_matrix(
    F: torch.Tensor,
    topk_indices: torch.Tensor,
    blur_strategy: str = 'mean'
) -> torch.Tensor:
    """
    Generate a bokeh-style matrix in feature space by restoring top-k patch embeddings.

    Args:
        F (torch.Tensor): Original feature matrix [N, D].
        topk_indices (torch.Tensor): Indices of top-k salient patches.
        blur_strategy (str): 'mean' or 'zero' for the neutral embedding.

    Returns:
        torch.Tensor: Feature matrix [N, D] with only top-k restored.
    """
    N, D = F.shape
    if blur_strategy == 'mean':
        F_blur = F.mean(dim=0, keepdim=True)  # [1, D]
    elif blur_strategy == 'zero':
        F_blur = torch.zeros((1, D), dtype=F.dtype, device=F.device)
    else:
        raise ValueError("Invalid blur_strategy. Use 'mean' or 'zero'.")

    F_bokeh = F_blur.repeat(N, 1)
    F_bokeh[topk_indices] = F[topk_indices]
    return F_bokeh


def entropy_proxy_frobenius(F_bokeh: torch.Tensor) -> float:
    """Compute Frobenius norm as entropy proxy."""
    return torch.norm(F_bokeh, p='fro').item()


def entropy_proxy_variance(F_bokeh: torch.Tensor) -> float:
    """Compute total feature variance as entropy proxy."""
    return torch.var(F_bokeh, dim=0).sum().item()


def entropy_proxy_pca(F_bokeh: torch.Tensor, k: int = None) -> float:
    """
    Compute PCA-based entropy (spectral entropy) as proxy.

    Args:
        F_bokeh (torch.Tensor): Feature matrix [N, D].
        k (int): Number of principal components to consider. If None, use full rank.

    Returns:
        float: Spectral entropy (Shannon entropy of normalized eigenvalues).
    """
    F_centered = F_bokeh - F_bokeh.mean(dim=0, keepdim=True)
    q = k if k is not None else min(F_bokeh.shape)
    U, S, V = torch.pca_lowrank(F_centered, q=q)

    p = S / S.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
    return entropy


def compute_feature_entropy_metrics(
    F: torch.Tensor,
    topk_indices: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute all three entropy proxies from a bokeh-style feature matrix.

    Args:
        F (torch.Tensor): Original feature matrix [N, D].
        topk_indices (torch.Tensor): Indices of important patches to retain.

    Returns:
        Tuple of Frobenius norm, total variance, and PCA entropy.
    """
    F_bokeh = compute_feature_bokeh_matrix(F, topk_indices)
    frob = entropy_proxy_frobenius(F_bokeh)
    var = entropy_proxy_variance(F_bokeh)
    pca = entropy_proxy_pca(F_bokeh)
    return frob, var, pca

F = torch.randn(1000, 512)  # 1000 patches, 512-dim embeddings
topk_indices = torch.tensor([10, 25, 77, 144, 301])  # top-5 important patches

frob, var, pca = compute_feature_entropy_metrics(F, topk_indices)
print("Frobenius norm:", frob)
print("Total variance:", var)
print("PCA entropy:", pca)
