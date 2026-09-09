"""Spatial prediction + uncertainty maps for ECCV paper.

Creates qualitative figures showing:
  - VIIRS false-color imagery
  - Predicted cloud geometry (cloud top height) at all 64x64 pixels
  - Ground truth at EarthCARE track positions
  - Prediction interval width (uncertainty) at all pixels
  - Global test patch coverage map

Usage:
    python -u 61_spatial_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from config import OUTPUT_DIR, COLOC_DIR
from models.convnext_unet import ConvNextUNet

DENSE_DIR = COLOC_DIR / "npy_dense"
PAPER_DIR = Path("figures/eccv")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

N_TARGETS = 8
TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]

# Selected patches: diverse cloud types, 0% fill values, high spatial variation
SELECTED_PATCHES = [
    (8602,  "Mixed (11.5°N, Indian Ocean)"),
    (3128,  "Deep Conv. (-5.7°S, Indonesia)"),
    (13542, "Tropical (20.0°N, Sudan)"),
    (6242,  "Mid-level (-43.9°S, S. Ocean)"),
]

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def level_to_km(level):
    """Convert pressure level index to altitude in km."""
    return 16.2 - (level - 69) * (16.2 - 0.6) / 158


def load_ensemble(device):
    """Load 5 quantile models for G-QE ensemble."""
    tags = ["G-Q", "G-Q2", "G-Q3", "G-Q4", "G-Q5"]
    models = []
    geo_mean = geo_std = era5_mean = era5_std = None

    for tag in tags:
        ckpt_path = OUTPUT_DIR / "models" / f"geom_{tag}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if geo_mean is None:
            geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
            geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
            era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
            era5_std = np.array(ckpt["era5_std"], dtype=np.float32)

        model = ConvNextUNet(
            in_channels=10, out_channels=N_TARGETS,
            base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
            era5_dim=ckpt["era5_dim"],
            quantile_mode=True, physics_head=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models, geo_mean, geo_std, era5_mean, era5_std


def predict_ensemble(models, patch, era5, era5_valid, geo_mean, geo_std, device):
    """Run G-QE ensemble prediction on a single patch.

    Returns: pred (8, 3, 64, 64) in physical units — median of 5 members.
    """
    patch_t = torch.from_numpy(patch[None]).to(device)
    era5_t = torch.from_numpy(era5[None]).to(device)
    era5v_t = torch.from_numpy(np.array([era5_valid])).to(device)

    preds = []
    with torch.no_grad():
        for model in models:
            with autocast("cuda", dtype=torch.bfloat16):
                p = model(patch_t, era5_t, era5v_t)
            preds.append(p.float().cpu().numpy()[0])  # (8, 3, 64, 64)

    # Average across members
    pred = np.mean(preds, axis=0)  # (8, 3, 64, 64)

    # Denormalize to physical units
    for t in range(N_TARGETS):
        pred[t] = pred[t] * geo_std[t] + geo_mean[t]

    return pred


def fig_spatial_examples():
    """Main spatial figure: 4 rows × 4 cols showing predictions + uncertainty."""
    device = torch.device("cuda")
    print("  Loading G-QE ensemble (5 models)...")
    models, geo_mean, geo_std, era5_mean, era5_std = load_ensemble(device)

    # Load test data
    patches = np.load(DENSE_DIR / "test_patches.npy")
    geometry = np.load(DENSE_DIR / "test_geometry.npy")
    positions = np.load(DENSE_DIR / "test_positions.npy")
    n_profiles = np.load(DENSE_DIR / "test_n_profiles.npy")

    era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
    era5_flat = era5_raw.reshape(len(era5_raw), -1)
    era5_flat = (era5_flat - era5_mean) / era5_std
    era5_flat[era5_valid < 0.5] = 0.0

    n_examples = len(SELECTED_PATCHES)
    fig, axes = plt.subplots(n_examples, 4, figsize=(7.0, n_examples * 1.8))

    # Target to plot: cloud_top (index 1) — most intuitive
    target_idx = 1  # cloud_top
    target_name = "Cloud Top Height"

    for row, (patch_idx, label) in enumerate(SELECTED_PATCHES):
        print(f"  Processing patch {patch_idx}: {label}")

        # Run ensemble prediction
        pred = predict_ensemble(
            models, patches[patch_idx], era5_flat[patch_idx],
            era5_valid[patch_idx], geo_mean, geo_std, device
        )  # (8, 3, 64, 64)

        # Extract cloud top: median prediction and interval width
        cloud_top_med = pred[target_idx, 1]  # (64, 64) median
        cloud_top_lo = pred[target_idx, 0]   # (64, 64) q10
        cloud_top_hi = pred[target_idx, 2]   # (64, 64) q90
        interval_width = cloud_top_hi - cloud_top_lo  # (64, 64)

        # Convert to km
        cloud_top_km = level_to_km(cloud_top_med)
        interval_width_km = interval_width * (16.2 - 0.6) / 158  # levels → km

        # Ground truth at track positions
        n = int(n_profiles[patch_idx])
        gt_rows = positions[patch_idx, :n, 0].astype(int)
        gt_cols = positions[patch_idx, :n, 1].astype(int)
        valid = (gt_rows >= 0) & (gt_rows < 64) & (gt_cols >= 0) & (gt_cols < 64)
        gt_rows, gt_cols = gt_rows[valid], gt_cols[valid]
        gt_top = geometry[patch_idx, :n, target_idx][valid]  # raw level index
        gt_top_km = level_to_km(gt_top)

        # (a) VIIRS false color: M15 thermal BT (mask fill values)
        ax = axes[row, 0]
        bt = patches[patch_idx, 3].copy()  # M15 (normalized)
        bt[bt < -10] = np.nan  # mask fill values
        ax.imshow(bt, cmap="gray_r", interpolation="nearest")
        ax.set_title("VIIRS (11 μm BT)" if row == 0 else "")
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        # Per-patch color range for cloud top
        all_top = np.concatenate([cloud_top_km.ravel(), gt_top_km])
        vmin = np.percentile(all_top, 2)
        vmax = np.percentile(all_top, 98)
        vpad = max((vmax - vmin) * 0.1, 0.5)
        vmin -= vpad
        vmax += vpad

        # (b) Predicted cloud top height (full 64×64)
        ax = axes[row, 1]
        im = ax.imshow(cloud_top_km, cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"Pred. {target_name} (km)" if row == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, format="%.0f")

        # (c) Predicted + GT track overlay
        ax = axes[row, 2]
        ax.imshow(cloud_top_km, cmap="viridis", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        # Overlay GT as colored scatter with red edges
        sc = ax.scatter(gt_cols, gt_rows, c=gt_top_km, cmap="viridis",
                       vmin=vmin, vmax=vmax, s=8, edgecolors="red",
                       linewidths=0.6, zorder=5)
        ax.set_title("Pred. + GT Track" if row == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])

        # (d) Interval width (uncertainty)
        ax = axes[row, 3]
        im_unc = ax.imshow(interval_width_km, cmap="magma",
                          interpolation="nearest")
        ax.set_title("Interval Width (km)" if row == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im_unc, ax=ax, shrink=0.8, pad=0.02, format="%.1f")

    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(PAPER_DIR / "spatial_predictions.pdf")
    fig.savefig(PAPER_DIR / "spatial_predictions.png")
    plt.close(fig)
    print(f"  Saved spatial_predictions.pdf")


def fig_global_predictions():
    """Global map of G-QE predicted cloud top height + uncertainty at all pixels.

    Runs full ensemble inference on all 14K test patches, then plots
    mean prediction per patch on a Robinson projection.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    device = torch.device("cuda")

    # Check for cached predictions
    cache_path = OUTPUT_DIR / "global_predictions_cache.npz"
    if cache_path.exists():
        print("  Loading cached global predictions...")
        cache = np.load(cache_path)
        patch_top_km = cache["patch_top_km"]
        patch_width_km = cache["patch_width_km"]
        patch_iwp = cache["patch_iwp"]
    else:
        print("  Loading G-QE ensemble (5 models)...")
        models, geo_mean, geo_std, era5_mean, era5_std = load_ensemble(device)

        # Load test data
        patches = np.load(DENSE_DIR / "test_patches.npy")
        era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
        nan_mask = np.isnan(era5_raw)
        era5_raw[nan_mask] = 0.0
        era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
        era5_flat = era5_raw.reshape(len(era5_raw), -1)
        era5_flat = (era5_flat - era5_mean) / era5_std
        era5_flat[era5_valid < 0.5] = 0.0

        N = len(patches)
        batch_size = 32
        IDX_TOP = 1
        IDX_IWP = 7

        patch_top_km = np.zeros(N, dtype=np.float32)
        patch_width_km = np.zeros(N, dtype=np.float32)
        patch_iwp = np.zeros(N, dtype=np.float32)

        print(f"  Running G-QE on {N:,} test patches...")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            p_batch = torch.from_numpy(patches[start:end]).to(device)
            e_batch = torch.from_numpy(era5_flat[start:end]).to(device)
            ev_batch = torch.from_numpy(era5_valid[start:end]).to(device)

            # Ensemble average
            ens_preds = []
            with torch.no_grad():
                for model in models:
                    with autocast("cuda", dtype=torch.bfloat16):
                        pred = model(p_batch, e_batch, ev_batch)
                    ens_preds.append(pred.float().cpu().numpy())
            pred = np.mean(ens_preds, axis=0)  # (B, 8, 3, 64, 64)

            for i in range(end - start):
                # Cloud top: median prediction, denormalized to level, then to km
                top_med = pred[i, IDX_TOP, 1] * geo_std[IDX_TOP] + geo_mean[IDX_TOP]  # (64,64) levels
                top_km = level_to_km(top_med)
                patch_top_km[start + i] = np.mean(top_km)

                # Interval width for cloud top
                top_hi = pred[i, IDX_TOP, 2] * geo_std[IDX_TOP] + geo_mean[IDX_TOP]
                top_lo = pred[i, IDX_TOP, 0] * geo_std[IDX_TOP] + geo_mean[IDX_TOP]
                width_levels = top_hi - top_lo
                patch_width_km[start + i] = np.mean(width_levels) * (16.2 - 0.6) / 158

                # IWP: median prediction, denormalized
                iwp_med = pred[i, IDX_IWP, 1] * geo_std[IDX_IWP] + geo_mean[IDX_IWP]
                patch_iwp[start + i] = np.mean(iwp_med)

            if (start + end - start) % 2000 < batch_size:
                print(f"    [{end:,}/{N:,}]")

        np.savez(cache_path, patch_top_km=patch_top_km,
                 patch_width_km=patch_width_km, patch_iwp=patch_iwp)
        print(f"  Cached to {cache_path}")

    lats = np.load(DENSE_DIR / "test_lat.npy")
    lons = np.load(DENSE_DIR / "test_lon.npy")

    # Figure: 2 panels — (a) Cloud Top Height, (b) Prediction Uncertainty
    proj = ccrs.Robinson()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                    subplot_kw={"projection": proj})

    for ax in (ax1, ax2):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#F5F5F5", edgecolor="none", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888", zorder=2)
        ax.gridlines(draw_labels=False, linewidth=0.15, color="gray", alpha=0.4)

    # (a) Cloud Top Height
    sc1 = ax1.scatter(lons, lats, c=patch_top_km, s=0.8, cmap="viridis",
                      vmin=4, vmax=18, alpha=0.7,
                      transform=ccrs.PlateCarree(), rasterized=True, zorder=3)
    ax1.set_title("(a) Predicted Cloud Top Height", fontsize=9)
    cb1 = fig.colorbar(sc1, ax=ax1, orientation="horizontal", pad=0.02,
                       shrink=0.8, aspect=30)
    cb1.set_label("km", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    # (b) Prediction Interval Width (uncertainty)
    sc2 = ax2.scatter(lons, lats, c=patch_width_km, s=0.8, cmap="magma",
                      vmin=1, vmax=8, alpha=0.7,
                      transform=ccrs.PlateCarree(), rasterized=True, zorder=3)
    ax2.set_title("(b) Prediction Uncertainty", fontsize=9)
    cb2 = fig.colorbar(sc2, ax=ax2, orientation="horizontal", pad=0.02,
                       shrink=0.8, aspect=30)
    cb2.set_label("Interval Width (km)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=0.5)
    fig.savefig(PAPER_DIR / "global_predictions.pdf")
    fig.savefig(PAPER_DIR / "global_predictions.png")
    plt.close(fig)
    print(f"  Saved global_predictions.pdf")


if __name__ == "__main__":
    print("Generating spatial figures...\n")
    fig_spatial_examples()
    fig_global_predictions()
    print("\nDone!")
