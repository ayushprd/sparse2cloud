"""Cross-track dense uncertainty analysis.

Key question: Does the model's uncertainty change with distance from the
EarthCARE track? If interval widths are uniform across all distances, the
model's uncertainty is based on spectral features, not spatial proximity
to supervision.

Analyses:
  (a) Interval width at ALL 64x64 pixels per patch — no ground truth needed.
      Bin by distance to nearest supervised pixel. Show width is constant.
  (b) R² at held-out supervised pixels using 50/50 split — proves retrieval
      quality doesn't degrade with distance.
  (c) PICP at held-out pixels — coverage is maintained across distance.

Usage:
    python -u 58_crosstrack_uncertainty.py [--method G-Q]
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import r2_score
from pathlib import Path

from config import OUTPUT_DIR, COLOC_DIR
from models.convnext_unet import ConvNextUNet

DENSE_DIR = COLOC_DIR / "npy_dense"
N_TARGETS = 8
TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]


def load_model(ckpt_path, device):
    """Load quantile model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
    geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
    era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
    era5_std = np.array(ckpt["era5_std"], dtype=np.float32)

    model = ConvNextUNet(
        in_channels=10, out_channels=N_TARGETS,
        base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
        era5_dim=ckpt["era5_dim"],
        quantile_mode=True,
        physics_head=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, geo_mean, geo_std, era5_mean, era5_std


def run_crosstrack_analysis(method="G-Q"):
    """Run full cross-track uncertainty analysis."""
    device = torch.device("cuda")
    ckpt_path = OUTPUT_DIR / "models" / f"geom_{method}.pt"
    print(f"  Loading model: {ckpt_path}")
    model, geo_mean, geo_std, era5_mean, era5_std = load_model(ckpt_path, device)

    # Load test data
    print("  Loading test data...")
    patches = np.load(DENSE_DIR / "test_patches.npy")
    geometry = np.load(DENSE_DIR / "test_geometry.npy")  # (N, 64, 8)
    positions = np.load(DENSE_DIR / "test_positions.npy")  # (N, 64, 2)
    n_profiles = np.load(DENSE_DIR / "test_n_profiles.npy")

    # ERA5
    era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
    era5_flat = era5_raw.reshape(len(era5_raw), -1)
    era5_flat = (era5_flat - era5_mean) / era5_std
    era5_flat[era5_valid < 0.5] = 0.0

    N = len(patches)
    batch_size = 32
    rng = np.random.RandomState(42)

    # Accumulators
    # (a) All-pixel interval widths binned by distance
    dist_bins = [0, 1, 2, 3, 5, 8, 12, 20, 32, 64]
    allpx_width_sums = {t: np.zeros(len(dist_bins) - 1) for t in range(N_TARGETS)}
    allpx_counts = np.zeros(len(dist_bins) - 1, dtype=np.int64)

    # (b,c) Held-out supervised pixel metrics
    heldout_pred = []   # (M, 8, 3) quantile predictions
    heldout_true = []   # (M, 8) ground truth
    heldout_dist = []   # (M,) distance to nearest retained pixel

    print(f"  Processing {N:,} test patches...")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        patches_b = torch.from_numpy(patches[start:end]).to(device)
        era5_b = torch.from_numpy(era5_flat[start:end]).to(device)
        era5v_b = torch.from_numpy(era5_valid[start:end]).to(device)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                pred_b = model(patches_b, era5_b, era5v_b)
            pred_b = pred_b.float().cpu().numpy()  # (B, 8, 3, 64, 64)

        for i in range(B):
            idx = start + i
            n_prof = int(n_profiles[idx])
            if n_prof < 4:
                continue

            rows = positions[idx, :n_prof, 0].astype(np.int64)
            cols = positions[idx, :n_prof, 1].astype(np.int64)
            valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
            rows, cols = rows[valid], cols[valid]
            n_valid = len(rows)
            if n_valid < 4:
                continue

            # Build supervision mask for distance computation
            sup_mask = np.zeros((64, 64), dtype=np.float32)
            sup_mask[rows, cols] = 1.0
            dist_map = distance_transform_edt(1 - sup_mask)  # (64, 64)

            # (a) All-pixel interval widths (vectorized)
            # pred_b[i] is (8, 3, 64, 64): channels, quantiles, H, W
            bin_idx_map = np.digitize(dist_map.ravel(), dist_bins) - 1  # (4096,)
            bin_idx_map = np.clip(bin_idx_map, 0, len(dist_bins) - 2)
            for b_idx in range(len(dist_bins) - 1):
                px_mask = (bin_idx_map == b_idx)
                n_px = int(px_mask.sum())
                if n_px == 0:
                    continue
                allpx_counts[b_idx] += n_px
                for t in range(N_TARGETS):
                    widths = (pred_b[i, t, 2].ravel()[px_mask] - pred_b[i, t, 0].ravel()[px_mask]) * geo_std[t]
                    allpx_width_sums[t][b_idx] += float(widths.sum())

            # (b,c) 50/50 split for held-out R² and PICP
            perm = rng.permutation(n_valid)
            n_retain = n_valid // 2
            retain_idx = perm[:n_retain]
            heldout_idx = perm[n_retain:]

            if len(retain_idx) < 2 or len(heldout_idx) < 2:
                continue

            # Distance from retained pixels only
            retain_mask = np.zeros((64, 64), dtype=np.float32)
            retain_mask[rows[retain_idx], cols[retain_idx]] = 1.0
            dist_map_split = distance_transform_edt(1 - retain_mask)

            for j in heldout_idx:
                r, c = rows[j], cols[j]
                p = pred_b[i, :, :, r, c] * geo_std[:, None] + geo_mean[:, None]  # (8, 3)
                geo_vals = geometry[idx, j]  # (8,) raw geometry values
                heldout_pred.append(p)
                heldout_true.append(geo_vals)
                heldout_dist.append(dist_map_split[r, c])

        if (start + B) % 2000 < batch_size:
            print(f"    [{start + B:,}/{N:,}] "
                  f"allpx={int(allpx_counts.sum()):,}, "
                  f"heldout={len(heldout_pred):,}")

    # Compile results
    print(f"\n  Total all-pixel samples: {int(allpx_counts.sum()):,}")
    print(f"  Total held-out samples: {len(heldout_pred):,}")

    heldout_pred = np.stack(heldout_pred)   # (M, 8, 3)
    heldout_true = np.stack(heldout_true)   # (M, 8)
    heldout_dist = np.array(heldout_dist)   # (M,)

    # Load CQR thresholds from saved test data
    cqr_path = OUTPUT_DIR / f"cqr_test_data_{method}.npz"
    if cqr_path.exists():
        cqr_data = np.load(cqr_path)
        q_hat = cqr_data["q_hat"]  # (8,) CQR thresholds
        print(f"  CQR thresholds loaded from {cqr_path}")
    else:
        q_hat = np.zeros(N_TARGETS)
        print(f"  WARNING: No CQR data, using q_hat=0")

    results = {"method": method, "dist_bins": dist_bins}

    # (a) All-pixel interval width vs distance
    print(f"\n  {'='*70}")
    print(f"  (a) Mean Interval Width at ALL Pixels vs Distance to Track")
    print(f"  {'='*70}")
    print(f"  {'Bin':>8s}  {'Count':>10s}", end="")
    for name in TARGET_NAMES[:5]:
        print(f"  {name:>10s}", end="")
    print()

    allpx_results = {}
    for b in range(len(dist_bins) - 1):
        lo, hi = dist_bins[b], dist_bins[b + 1]
        n = int(allpx_counts[b])
        bin_key = f"{lo}-{hi}"
        if n == 0:
            continue
        mean_widths = {TARGET_NAMES[t]: float(allpx_width_sums[t][b] / n) for t in range(N_TARGETS)}
        allpx_results[bin_key] = {"n": n, "mean_width": mean_widths}
        print(f"  {bin_key:>8s}  {n:>10,}", end="")
        for t in range(5):
            print(f"  {mean_widths[TARGET_NAMES[t]]:>10.2f}", end="")
        print()

    results["allpixel_width"] = allpx_results

    # (b) R² vs distance (50/50 split)
    print(f"\n  {'='*70}")
    print(f"  (b) R² at Held-out Pixels vs Distance to Retained Supervision")
    print(f"  {'='*70}")
    print(f"  {'Bin':>8s}  {'Count':>8s}  {'mean_R²':>8s}", end="")
    for name in TARGET_NAMES[:5]:
        print(f"  {name:>10s}", end="")
    print()

    heldout_results = {}
    for b in range(len(dist_bins) - 1):
        lo, hi = dist_bins[b], dist_bins[b + 1]
        mask = (heldout_dist >= lo) & (heldout_dist < hi)
        n = int(mask.sum())
        bin_key = f"{lo}-{hi}"
        if n < 100:
            continue

        bin_data = {"n": n}
        r2_per_target = {}
        picp_per_target = {}
        mpiw_per_target = {}
        for t, name in enumerate(TARGET_NAMES):
            y = heldout_true[mask, t]
            q_med = heldout_pred[mask, t, 1]
            q_lo = heldout_pred[mask, t, 0] - q_hat[t]
            q_hi = heldout_pred[mask, t, 2] + q_hat[t]
            r2_per_target[name] = float(r2_score(y, q_med))
            picp_per_target[name] = float(((y >= q_lo) & (y <= q_hi)).mean())
            mpiw_per_target[name] = float((q_hi - q_lo).mean())

        bin_data["r2"] = r2_per_target
        bin_data["picp"] = picp_per_target
        bin_data["mpiw"] = mpiw_per_target
        bin_data["mean_r2"] = float(np.mean(list(r2_per_target.values())))
        bin_data["mean_picp"] = float(np.mean(list(picp_per_target.values())))
        bin_data["mean_mpiw"] = float(np.mean(list(mpiw_per_target.values())))

        heldout_results[bin_key] = bin_data

        print(f"  {bin_key:>8s}  {n:>8,}  {bin_data['mean_r2']:>8.4f}", end="")
        for t_name in TARGET_NAMES[:5]:
            print(f"  {r2_per_target[t_name]:>10.4f}", end="")
        print()

    results["heldout"] = heldout_results

    # (c) PICP vs distance
    print(f"\n  {'='*70}")
    print(f"  (c) PICP at Held-out Pixels vs Distance")
    print(f"  {'='*70}")
    print(f"  {'Bin':>8s}  {'Count':>8s}  {'mean_PICP':>10s}  {'mean_MPIW':>10s}")
    for bin_key, bd in heldout_results.items():
        print(f"  {bin_key:>8s}  {bd['n']:>8,}  {bd['mean_picp']:>10.4f}  {bd['mean_mpiw']:>10.2f}")

    # Save
    out_path = OUTPUT_DIR / "results_crosstrack_dense.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="G-Q",
                        help="Model tag to evaluate (default: G-Q)")
    args = parser.parse_args()
    run_crosstrack_analysis(args.method)


if __name__ == "__main__":
    main()
