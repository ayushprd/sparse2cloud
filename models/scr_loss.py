"""Spectral Consistency Regularization (SCR) loss.

Enforces that spectrally similar pixels produce similar predictions,
propagating supervision from labeled to unlabeled pixels.

Three variants:
  1. SCR-raw:     similarity in raw input feature space
  2. SCR-encoder: similarity in learned encoder feature space
  3. SCR-pseudo:  kNN regression pseudo-labels from supervised pixels

All operate on flattened pixel collections within a mini-batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distances(a, b):
    """Compute pairwise squared L2 distances between rows of a and b.

    Args:
        a: (N, D), b: (M, D)
    Returns:
        (N, M) squared distances
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T
    aa = (a * a).sum(dim=1, keepdim=True)  # (N, 1)
    bb = (b * b).sum(dim=1, keepdim=True)  # (M, 1)
    dist = aa + bb.T - 2.0 * a @ b.T      # (N, M)
    return dist.clamp(min=0.0)


def _topk_neighbors(dists, k):
    """Find k nearest neighbors (excluding self).

    Args:
        dists: (N, N) pairwise distances
        k: number of neighbors
    Returns:
        indices: (N, k) neighbor indices
        weights: (N, k) inverse-distance weights (unnormalized)
    """
    # Set self-distance to inf to exclude
    dists = dists.clone()
    dists.fill_diagonal_(float('inf'))
    k = min(k, dists.shape[1] - 1)
    topk_dists, topk_idx = dists.topk(k, dim=1, largest=False)
    return topk_idx, topk_dists


class SCRLoss(nn.Module):
    """Spectral Consistency Regularization loss.

    For each pixel in a mini-batch, finds spectrally similar pixels and
    enforces their predictions to be consistent.

    Args:
        sigma: bandwidth for Gaussian similarity kernel
        k: number of nearest neighbors per pixel
        max_pixels: max pixels to sample per batch (memory control)
    """
    def __init__(self, sigma=1.0, k=16, max_pixels=2048):
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.max_pixels = max_pixels

    def forward(self, pred, features, mask=None):
        """Compute SCR loss.

        Args:
            pred: (B, C, H, W) model predictions (PCA coefficients)
            features: (B, D, H, W) spectral features for similarity computation
            mask: (B, H, W) optional; if provided, only uses pixels where mask > 0.5
                  for computing similarity targets (supervised pixels), but applies
                  consistency to ALL pixels

        Returns:
            loss: scalar SCR loss
        """
        B, C, H, W = pred.shape
        D = features.shape[1]

        # Flatten spatial dims
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)        # (B*H*W, C)
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, D)    # (B*H*W, D)

        # Subsample if too many pixels
        N = pred_flat.shape[0]
        if N > self.max_pixels:
            idx = torch.randperm(N, device=pred.device)[:self.max_pixels]
            pred_flat = pred_flat[idx]
            feat_flat = feat_flat[idx]
            N = self.max_pixels

        # Compute pairwise distances in feature space
        dists = _pairwise_distances(feat_flat, feat_flat)  # (N, N)

        # Find k nearest spectral neighbors
        nn_idx, nn_dists = _topk_neighbors(dists, self.k)  # (N, k), (N, k)

        # Gaussian similarity weights
        weights = torch.exp(-nn_dists / (2.0 * self.sigma ** 2))  # (N, k)

        # Gather neighbor predictions
        nn_preds = pred_flat[nn_idx]  # (N, k, C)

        # Consistency loss: weighted MSE between each pixel and its neighbors
        diff = pred_flat.unsqueeze(1) - nn_preds  # (N, k, C)
        weighted_mse = weights.unsqueeze(-1) * diff.pow(2)  # (N, k, C)

        # Normalize by total weight
        loss = weighted_mse.sum() / (weights.sum() * C + 1e-8)

        return loss


class SCRPseudoLabelLoss(nn.Module):
    """SCR via kNN pseudo-labels from supervised pixels.

    For each unsupervised pixel, finds k nearest supervised pixels in
    spectral feature space and uses their weighted average prediction
    as a pseudo-label.

    Args:
        sigma: bandwidth for Gaussian similarity kernel
        k: number of supervised neighbors
        max_unsup: max unsupervised pixels to process per batch
    """
    def __init__(self, sigma=1.0, k=16, max_unsup=2048):
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.max_unsup = max_unsup

    def forward(self, pred, features, mask, target):
        """Compute pseudo-label SCR loss.

        Args:
            pred: (B, C, H, W) model predictions
            features: (B, D, H, W) spectral features
            mask: (B, H, W) supervision mask (1 = supervised, 0 = unsupervised)
            target: (B, C, H, W) targets (only valid where mask=1)

        Returns:
            loss: scalar pseudo-label consistency loss
        """
        B, C, H, W = pred.shape
        D = features.shape[1]

        # Separate supervised and unsupervised pixels
        mask_flat = mask.reshape(-1) > 0.5  # (B*H*W,)
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)      # (B*H*W, C)
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        tgt_flat = target.permute(0, 2, 3, 1).reshape(-1, C)     # (B*H*W, C)

        sup_feat = feat_flat[mask_flat]    # (N_sup, D)
        sup_tgt = tgt_flat[mask_flat]      # (N_sup, C)
        unsup_feat = feat_flat[~mask_flat] # (N_unsup, D)
        unsup_pred = pred_flat[~mask_flat] # (N_unsup, C)

        if sup_feat.shape[0] < 2 or unsup_feat.shape[0] < 1:
            return torch.tensor(0.0, device=pred.device)

        # Subsample unsupervised pixels if too many
        N_unsup = unsup_feat.shape[0]
        if N_unsup > self.max_unsup:
            idx = torch.randperm(N_unsup, device=pred.device)[:self.max_unsup]
            unsup_feat = unsup_feat[idx]
            unsup_pred = unsup_pred[idx]
            N_unsup = self.max_unsup

        # Distances from unsupervised to supervised pixels
        dists = _pairwise_distances(unsup_feat, sup_feat)  # (N_unsup, N_sup)

        # Find k nearest supervised neighbors
        k = min(self.k, sup_feat.shape[0])
        topk_dists, topk_idx = dists.topk(k, dim=1, largest=False)  # (N_unsup, k)

        # Gaussian weights
        weights = torch.exp(-topk_dists / (2.0 * self.sigma ** 2))  # (N_unsup, k)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize

        # Pseudo-labels: weighted average of supervised targets
        nn_targets = sup_tgt[topk_idx]  # (N_unsup, k, C)
        pseudo_labels = (weights.unsqueeze(-1) * nn_targets).sum(dim=1)  # (N_unsup, C)

        # MSE between model prediction and pseudo-label (detached)
        loss = F.mse_loss(unsup_pred, pseudo_labels.detach())

        return loss
