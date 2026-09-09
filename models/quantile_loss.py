"""Quantile regression losses and physics-informed ordering constraints.

Used for Conformalized Quantile Regression (CQR) on cloud geometry targets.
"""
import torch
import torch.nn.functional as F


def pinball_loss(pred, target, mask, quantiles=(0.1, 0.5, 0.9)):
    """Masked pinball (quantile) loss for dense prediction.

    Args:
        pred: (B, T, Q, H, W) predicted quantiles
        target: (B, T, H, W) true values at supervised pixels
        mask: (B, H, W) supervision mask (1 = supervised)
        quantiles: tuple of quantile levels matching dim Q

    Returns:
        scalar loss
    """
    B, T, Q, H, W = pred.shape
    target_exp = target.unsqueeze(2)          # (B, T, 1, H, W)
    mask_exp = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)

    residual = target_exp - pred  # (B, T, Q, H, W)
    tau = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, Q, 1, 1)

    loss = torch.where(residual >= 0, tau * residual, (tau - 1.0) * residual)
    loss = loss * mask_exp

    n_sup = mask.sum() * T * Q + 1e-8
    return loss.sum() / n_sup


def ordering_loss(pred, target_idx, quantiles=(0.1, 0.5, 0.9)):
    """Physics-informed ordering constraints as soft penalties.

    Enforces:
      1. Quantile monotonicity: q(tau_lo) <= q(tau_hi)
      2. Geometry ordering at median: cloud_base <= centroid <= cloud_top
      3. cloud_base <= peak_level <= cloud_top
      4. Thickness consistency: thickness ≈ cloud_top - cloud_base

    Args:
        pred: (B, T, Q, H, W) predicted quantiles
        target_idx: dict mapping target name → index
        quantiles: tuple of quantile levels

    Returns:
        scalar penalty
    """
    penalty = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    Q = len(quantiles)
    q_med = Q // 2  # median index

    # 1. Quantile monotonicity per target: q(tau_i) <= q(tau_{i+1})
    for q in range(Q - 1):
        violation = F.relu(pred[:, :, q, :, :] - pred[:, :, q + 1, :, :])
        penalty = penalty + violation.mean()

    # 2. Geometry ordering at median
    i = target_idx
    # cloud_base <= centroid
    penalty = penalty + F.relu(
        pred[:, i['cloud_base'], q_med] - pred[:, i['centroid'], q_med]).mean()
    # centroid <= cloud_top
    penalty = penalty + F.relu(
        pred[:, i['centroid'], q_med] - pred[:, i['cloud_top'], q_med]).mean()
    # cloud_base <= peak_level
    penalty = penalty + F.relu(
        pred[:, i['cloud_base'], q_med] - pred[:, i['peak_level'], q_med]).mean()
    # peak_level <= cloud_top
    penalty = penalty + F.relu(
        pred[:, i['peak_level'], q_med] - pred[:, i['cloud_top'], q_med]).mean()

    # 3. Thickness consistency: thickness ≈ top - base (at median)
    if 'thickness' in i:
        pred_thick = pred[:, i['thickness'], q_med]
        impl_thick = pred[:, i['cloud_top'], q_med] - pred[:, i['cloud_base'], q_med]
        penalty = penalty + F.mse_loss(pred_thick, impl_thick.detach())

    return penalty
