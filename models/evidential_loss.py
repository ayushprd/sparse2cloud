"""Deep Evidential Regression loss (Amini et al., NeurIPS 2020).

Predicts Normal-Inverse-Gamma parameters (gamma, nu, alpha, beta) per target.
Single forward pass gives mean + aleatoric + epistemic uncertainty.

Note: Known to produce overconfident estimates (Meinert et al., AAAI 2023).
Included as a baseline for comparison.
"""
import torch
import torch.nn.functional as F
import math


def parse_nig(pred, n_targets=8):
    """Parse NIG parameters from raw model output.

    Args:
        pred: (B, 4T, H, W)

    Returns:
        gamma: (B, T, H, W) mean
        nu: (B, T, H, W) virtual observations (> 0)
        alpha: (B, T, H, W) shape (> 1)
        beta: (B, T, H, W) rate (> 0)
    """
    T = n_targets
    gamma = pred[:, 0:T]
    nu = F.softplus(pred[:, T:2*T]) + 1e-6
    alpha = F.softplus(pred[:, 2*T:3*T]) + 1.0 + 1e-6
    beta = F.softplus(pred[:, 3*T:4*T]) + 1e-6
    return gamma, nu, alpha, beta


def evidential_loss(pred, target, mask, n_targets=8, coeff=0.01):
    """Deep Evidential Regression loss.

    Args:
        pred: (B, 4T, H, W) raw NIG parameters
        target: (B, T, H, W) true values
        mask: (B, H, W) supervision mask
        coeff: regularization coefficient

    Returns:
        scalar loss
    """
    gamma, nu, alpha, beta = parse_nig(pred, n_targets)

    omega = 2.0 * beta * (1.0 + nu)

    # NLL of Student-t
    nll = (0.5 * torch.log(math.pi / nu)
           - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

    # Evidence regularizer: penalize high evidence when prediction is wrong
    reg = torch.abs(target - gamma) * (2.0 * nu + alpha)

    loss = nll + coeff * reg

    mask_3d = mask.unsqueeze(1)
    n_sup = mask_3d.sum() * n_targets + 1e-8
    return (loss * mask_3d).sum() / n_sup
