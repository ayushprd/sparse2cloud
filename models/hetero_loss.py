"""Heteroscedastic regression with beta-NLL loss (Seitzer et al., ICLR 2022).

Predicts mean + log-variance per target. The beta-NLL fix prevents the model
from 'cheating' by inflating variance on hard examples.
"""
import torch
import torch.nn.functional as F


def beta_nll_loss(pred, target, mask, beta=0.5):
    """Beta-NLL loss for heteroscedastic regression.

    Args:
        pred: (B, 2T, H, W) first T channels = mean, last T = log_var
        target: (B, T, H, W) true values
        mask: (B, H, W) supervision mask
        beta: weighting exponent (0.5 recommended)

    Returns:
        scalar loss
    """
    T = target.shape[1]
    mean = pred[:, :T]
    log_var = pred[:, T:].clamp(-10, 10)
    var = torch.exp(log_var).clamp(min=1e-6)

    # Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / var)
    nll = 0.5 * (log_var + (target - mean) ** 2 / var)

    # Beta weighting: var.detach() stops gradient through the weight
    if beta > 0:
        nll = nll * (var.detach() ** beta)

    mask_3d = mask.unsqueeze(1)  # (B, 1, H, W)
    n_sup = mask_3d.sum() * T + 1e-8
    return (nll * mask_3d).sum() / n_sup
