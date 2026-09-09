"""Physics-Constrained Output Head for cloud geometry prediction.

Guarantees geometry ordering + quantile monotonicity by construction:
  - cloud_base <= centroid <= cloud_top
  - cloud_base <= peak_level <= cloud_top
  - thickness = cloud_top - cloud_base >= 0
  - q_lo <= q_med <= q_hi (per target)

Uses differentiable parameterization:
  7 raw primitives -> 8 constrained targets via softplus/sigmoid transforms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstrainedHead(nn.Module):
    """Output head that enforces cloud geometry ordering by construction.

    Parameterization (7 primitives -> 8 targets):
      cloud_base   = raw_base                             (free)
      thickness    = softplus(raw_thick)                   (>= 0)
      cloud_top    = cloud_base + thickness                (>= cloud_base)
      centroid     = cloud_base + sigmoid(raw_cent) * thickness  (in [base, top])
      peak_level   = cloud_base + sigmoid(raw_peak) * thickness  (in [base, top])
      core_iwc     = raw_core                              (free)
      mean_iwc     = raw_mean                              (free)
      log_iwp      = raw_iwp                               (free)

    For quantile mode, cumulative softplus ensures q_lo <= q_med <= q_hi:
      q_lo  = raw
      q_med = q_lo + softplus(delta_1)
      q_hi  = q_med + softplus(delta_2)

    Args:
        in_dim: input feature dimension from decoder
        quantile_mode: if True, output 3 quantiles per target
        n_quantiles: number of quantiles (default 3)
    """

    N_PRIMITIVES = 7  # base, thick, cent_frac, peak_frac, core, mean, iwp
    N_TARGETS = 8     # base, top, centroid, peak, thickness, core, mean, iwp

    # Target ordering in output
    IDX_BASE = 2       # cloud_base
    IDX_TOP = 1        # cloud_top
    IDX_CENTROID = 0   # centroid
    IDX_PEAK = 3       # peak_level
    IDX_THICKNESS = 4  # thickness
    IDX_CORE = 5       # core_iwc
    IDX_MEAN = 6       # mean_iwc
    IDX_IWP = 7        # log_iwp

    def __init__(self, in_dim, quantile_mode=False, n_quantiles=3):
        super().__init__()
        self.quantile_mode = quantile_mode
        self.n_quantiles = n_quantiles if quantile_mode else 1

        # In quantile mode: each primitive needs 3 raws for cumulative encoding
        # (base_q, delta_lo_to_med, delta_med_to_hi)
        n_raw = self.N_PRIMITIVES * (3 if quantile_mode else 1)

        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GroupNorm(1, in_dim),  # LayerNorm equivalent
            nn.GELU(),
            nn.Conv2d(in_dim, n_raw, 1),
        )

        # Initialize biases for physical plausibility
        self._init_biases()

    def _init_biases(self):
        """Set output biases so initial predictions are physically plausible."""
        conv = self.head[-1]  # final Conv2d
        nn.init.zeros_(conv.bias)

        with torch.no_grad():
            if self.quantile_mode:
                # 7 primitives × 3 (base_q, delta1, delta2) = 21 channels
                # thickness raw (primitive 1): bias so softplus ≈ 1.5 (mean thickness in z-space)
                conv.bias[1 * 3] = 1.5    # base_q for thickness
                conv.bias[1 * 3 + 1] = 0.5  # delta1 → softplus(0.5) ≈ 0.97
                conv.bias[1 * 3 + 2] = 0.5  # delta2 → softplus(0.5) ≈ 0.97
            else:
                # 7 channels: base, thick, cent_frac, peak_frac, core, mean, iwp
                conv.bias[1] = 1.5  # thickness raw

    def _cumulative_quantiles(self, raw_3):
        """Convert 3 raw values to ordered quantiles via cumulative softplus.

        Args:
            raw_3: (..., 3) tensor where dim=-1 is [base, delta1, delta2]

        Returns:
            (..., 3) tensor with [q_lo, q_med, q_hi], q_lo <= q_med <= q_hi
        """
        q_lo = raw_3[..., 0]
        q_med = q_lo + F.softplus(raw_3[..., 1])
        q_hi = q_med + F.softplus(raw_3[..., 2])
        return torch.stack([q_lo, q_med, q_hi], dim=-1)

    def _apply_physics(self, primitives):
        """Apply physics constraints to get 8 targets from 7 primitives.

        Args:
            primitives: (B, 7, H, W) or (B, 7, Q, H, W)

        Returns:
            (B, 8, H, W) or (B, 8, Q, H, W) constrained targets
        """
        # Unpack primitives
        raw_base = primitives[:, 0]      # cloud_base (free)
        raw_thick = primitives[:, 1]     # thickness via softplus
        raw_cent = primitives[:, 2]      # centroid fraction via sigmoid
        raw_peak = primitives[:, 3]      # peak fraction via sigmoid
        raw_core = primitives[:, 4]      # core_iwc (free)
        raw_mean = primitives[:, 5]      # mean_iwc (free)
        raw_iwp = primitives[:, 6]       # log_iwp (free)

        # Apply physics transforms
        cloud_base = raw_base
        thickness = F.softplus(raw_thick)
        cloud_top = cloud_base + thickness
        centroid = cloud_base + torch.sigmoid(raw_cent) * thickness
        peak_level = cloud_base + torch.sigmoid(raw_peak) * thickness

        # Assemble in target ordering
        targets = torch.stack([
            centroid,     # 0: centroid
            cloud_top,    # 1: cloud_top
            cloud_base,   # 2: cloud_base
            peak_level,   # 3: peak_level
            thickness,    # 4: thickness
            raw_core,     # 5: core_iwc
            raw_mean,     # 6: mean_iwc
            raw_iwp,      # 7: log_iwp
        ], dim=1)

        return targets

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, H, W) decoder features

        Returns:
            Point mode:    (B, 8, H, W)
            Quantile mode: (B, 8, 3, H, W)
        """
        raw = self.head(x)  # (B, n_raw, H, W)
        B, _, H, W = raw.shape

        if not self.quantile_mode:
            # Point mode: 7 raw → 8 targets
            primitives = raw  # (B, 7, H, W)
            return self._apply_physics(primitives)

        # Quantile mode: 21 raw → 7 primitives × 3 quantiles → 8 targets × 3 quantiles
        # Reshape: (B, 7, 3, H, W)
        raw = raw.view(B, self.N_PRIMITIVES, 3, H, W)

        # Apply cumulative softplus for quantile monotonicity per primitive
        # raw[..., :] is (B, 7, 3, H, W), treat last grouped dim as [base, d1, d2]
        # Need to permute for _cumulative_quantiles which expects (..., 3)
        raw_perm = raw.permute(0, 1, 3, 4, 2)  # (B, 7, H, W, 3)
        ordered = self._cumulative_quantiles(raw_perm)  # (B, 7, H, W, 3)
        ordered = ordered.permute(0, 1, 4, 2, 3)  # (B, 7, 3, H, W)

        # Apply physics per quantile
        results = []
        for q in range(3):
            primitives_q = ordered[:, :, q, :, :]  # (B, 7, H, W)
            targets_q = self._apply_physics(primitives_q)  # (B, 8, H, W)
            results.append(targets_q)

        # Stack: (B, 8, 3, H, W)
        return torch.stack(results, dim=2)
