"""Pre-extract cloud geometry targets from dense IWC profile data.

Computes 8 geometric properties per supervised pixel from 159-level profiles:
  centroid, cloud_top, cloud_base, peak_level, thickness, core_iwc, mean_iwc, log_iwp

Saves to npy_dense/{split}_geometry.npy (N, 64, 8) + geometry_stats.npz

Usage:
    python -u 48_extract_geometry_targets.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from config import COLOC_DIR, OUTPUT_DIR, LOG_IWC_EPS

DENSE_DIR = COLOC_DIR / "npy_dense"
LOG_FLOOR = np.log10(LOG_IWC_EPS)

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]


def compute_cloud_geometry(targets):
    """Compute 8 cloud geometry targets from 159-level log10(IWC) profiles.

    Args:
        targets: (N, 159) log10(IWC + eps) values

    Returns:
        (N, 8) array: [centroid, cloud_top, cloud_base, peak_level,
                        thickness, core_iwc, mean_iwc, log_iwp]
    """
    N, n_levels = targets.shape

    iwc_linear = 10.0 ** targets - LOG_IWC_EPS
    iwc_linear = np.maximum(iwc_linear, 0.0)

    ice_threshold = 10.0 ** (-3.5) - LOG_IWC_EPS
    ice_mask = iwc_linear > ice_threshold

    level_indices = np.arange(n_levels, dtype=np.float32)

    # Centroid: IWC-weighted mean level
    weights = iwc_linear * ice_mask
    weight_sum = weights.sum(axis=1, keepdims=True) + 1e-12
    centroid = (weights * level_indices).sum(axis=1) / weight_sum.squeeze()

    # Cloud top/base/peak via loop
    cloud_top = np.zeros(N, dtype=np.float32)
    cloud_base = np.full(N, 158.0, dtype=np.float32)
    peak_level = np.full(N, 79.0, dtype=np.float32)

    for i in range(N):
        ice_levels = np.where(ice_mask[i])[0]
        if len(ice_levels) > 0:
            cloud_top[i] = ice_levels[-1]
            cloud_base[i] = ice_levels[0]
            peak_level[i] = np.argmax(iwc_linear[i])

    thickness = cloud_top - cloud_base

    # Core IWC: mean log10(IWC) in upper troposphere (levels 1-41)
    core_iwc = targets[:, 1:42].mean(axis=1)

    # Mean IWC across full profile
    mean_iwc = targets.mean(axis=1)

    # Log IWP: log10(sum of linear IWC)
    log_iwp = np.log10(iwc_linear.sum(axis=1) + 1e-6)

    result = np.stack([
        centroid, cloud_top, cloud_base, peak_level,
        thickness, core_iwc, mean_iwc, log_iwp,
    ], axis=1).astype(np.float32)

    return result


def main():
    print("Extracting cloud geometry targets from dense profiles...")
    print(f"Targets: {TARGET_NAMES}")
    print()

    all_train_vals = []

    for split in ["train", "val", "test"]:
        targets = np.load(DENSE_DIR / f"{split}_targets.npy")  # (N, 64, 159)
        n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")
        N = len(targets)

        geometry = np.zeros((N, 64, 8), dtype=np.float32)

        for i in range(N):
            n = int(n_profiles[i])
            if n > 0:
                profiles = targets[i, :n]  # (n, 159)
                geom = compute_cloud_geometry(profiles)  # (n, 8)
                geometry[i, :n] = geom

        save_path = DENSE_DIR / f"{split}_geometry.npy"
        np.save(save_path, geometry)
        print(f"  {split}: {N:,} patches → {save_path} ({geometry.nbytes/1e6:.1f} MB)")

        # Collect supervised values for normalization stats
        if split == "train":
            for i in range(N):
                n = int(n_profiles[i])
                if n > 0:
                    all_train_vals.append(geometry[i, :n])

    # Compute normalization stats from training set
    all_vals = np.concatenate(all_train_vals, axis=0)  # (total_pixels, 8)
    geo_mean = all_vals.mean(axis=0).astype(np.float32)
    geo_std = all_vals.std(axis=0).astype(np.float32) + 1e-8

    stats_path = OUTPUT_DIR / "geometry_stats.npz"
    np.savez(stats_path,
             mean=geo_mean, std=geo_std,
             target_names=TARGET_NAMES)

    print(f"\nNormalization stats saved to {stats_path}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:>12s}: mean={geo_mean[i]:8.3f}, std={geo_std[i]:8.3f}")

    print(f"\nTotal training pixels: {len(all_vals):,}")
    print("Done!")


if __name__ == "__main__":
    main()
