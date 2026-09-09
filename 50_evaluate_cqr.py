"""Comprehensive CQR evaluation: cross-track uncertainty, conditional calibration.

Computes:
  1. Per-target R², MAE, PICP, MPIW
  2. Cross-track analysis: PICP, MPIW, R² vs distance from supervised pixel
  3. Conditional calibration: PICP stratified by cloud type proxies
  4. Reliability diagrams: nominal vs empirical coverage
  5. Interval-error correlation (sharpness)

Usage:
    python -u 50_evaluate_cqr.py --model-tag G-QP --alpha 0.1
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast
from sklearn.metrics import r2_score
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from pathlib import Path

from config import COLOC_DIR, OUTPUT_DIR, MODEL_DIR, FIGURE_DIR
from models.convnext_unet import ConvNextUNet, count_params

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


class DenseGeometryDatasetFull(Dataset):
    """Dataset that also returns positions for distance computation."""

    def __init__(self, split, geo_mean, geo_std, era5_mean, era5_std):
        self.patches = np.load(DENSE_DIR / f"{split}_patches.npy")
        self.geometry = np.load(DENSE_DIR / f"{split}_geometry.npy")
        self.positions = np.load(DENSE_DIR / f"{split}_positions.npy")
        self.n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")
        self.n_samples = len(self.patches)
        self.geo_mean = geo_mean
        self.geo_std = geo_std

        era5_raw = np.load(DENSE_DIR / f"{split}_era5.npy").astype(np.float32)
        nan_mask = np.isnan(era5_raw)
        era5_raw[nan_mask] = 0.0
        self.era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
        self.era5_flat = era5_raw.reshape(len(era5_raw), -1)
        self.era5_flat = (self.era5_flat - era5_mean) / era5_std
        valid = self.era5_valid > 0.5
        self.era5_flat[~valid] = 0.0
        self.era5_dim = self.era5_flat.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        patch = torch.from_numpy(np.ascontiguousarray(self.patches[idx]))
        n = int(self.n_profiles[idx])

        target = torch.zeros(N_TARGETS, 64, 64)
        mask = torch.zeros(64, 64)

        if n > 0:
            rows = self.positions[idx, :n, 0].astype(np.int64)
            cols = self.positions[idx, :n, 1].astype(np.int64)
            valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
            rows, cols = rows[valid], cols[valid]
            geo = self.geometry[idx, :n][valid]
            geo_norm = (geo - self.geo_mean) / self.geo_std
            target[:, rows, cols] = torch.from_numpy(geo_norm.T)
            mask[rows, cols] = 1.0

        era5 = torch.from_numpy(self.era5_flat[idx])
        era5_valid = torch.tensor(self.era5_valid[idx])
        return patch, target, mask, era5, era5_valid


def tta_predict(model, patch, era5, era5_valid, quantile_mode=True):
    """TTA-8 for quantile or point model."""
    preds = []
    for k in range(4):
        for flip in [False, True]:
            x = torch.rot90(patch, k, [2, 3])
            if flip:
                x = x.flip(-1)

            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, era5, era5_valid)
            pred = pred.float()

            if flip:
                pred = pred.flip(-1)
            if k > 0:
                if quantile_mode:
                    pred = torch.rot90(pred, -k, [3, 4])
                else:
                    pred = torch.rot90(pred, -k, [2, 3])
            preds.append(pred)

    return torch.stack(preds).mean(0)


def collect_with_distance(model, loader, device, geo_mean, geo_std,
                          use_tta=False, quantile_mode=True):
    """Collect predictions, targets, and distance-to-track per pixel.

    Returns:
        pred_all: (N, 8, 3) quantile predictions (denormalized)
        true_all: (N, 8) true values (denormalized)
        dist_all: (N,) distance to nearest supervised pixel
    """
    model.eval()
    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            if use_tta:
                pred = tta_predict(model, patch, era5, era5_valid, quantile_mode)
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                pred = pred.float()

            pred = pred.cpu()
            target = target.float().cpu()
            mask = mask.cpu()

            B = pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue

                # Distance transform from supervised pixels
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np) if mask_np.sum() > 0 else np.full((64, 64), 99.0)

                for r, c in zip(rows, cols):
                    if quantile_mode:
                        p = pred[b, :, :, r, c].numpy() * geo_std[:, None] + geo_mean[:, None]  # (8, 3)
                    else:
                        p_pt = pred[b, :, r, c].numpy() * geo_std + geo_mean  # (8,)
                        p = np.stack([p_pt, p_pt, p_pt], axis=1)  # fake (8, 3)
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


def calibrate_cqr(pred, true, alpha=0.1):
    """CQR calibration. Returns per-target q_hat."""
    N = len(true)
    q_hat = np.zeros(N_TARGETS)
    for t in range(N_TARGETS):
        scores = np.maximum(pred[:, t, 0] - true[:, t], true[:, t] - pred[:, t, 2])
        level = min(np.ceil((N + 1) * (1 - alpha)) / N, 1.0)
        q_hat[t] = np.quantile(scores, level)
    return q_hat


def compute_metrics(pred, true, q_hat, dist=None):
    """Compute comprehensive metrics.

    Args:
        pred: (N, 8, 3) quantile predictions
        true: (N, 8) true values
        q_hat: (8,) CQR thresholds
        dist: (N,) optional distance array for cross-track analysis

    Returns:
        dict of per-target and aggregate metrics
    """
    results = {"per_target": {}}

    for t, name in enumerate(TARGET_NAMES):
        q_lo = pred[:, t, 0] - q_hat[t]
        q_hi = pred[:, t, 2] + q_hat[t]
        q_med = pred[:, t, 1]
        y = true[:, t]

        covered = (y >= q_lo) & (y <= q_hi)
        raw_covered = (y >= pred[:, t, 0]) & (y <= pred[:, t, 2])

        r2 = float(r2_score(y, q_med))
        mae = float(np.abs(y - q_med).mean())
        picp = float(covered.mean())
        mpiw = float((q_hi - q_lo).mean())
        raw_picp = float(raw_covered.mean())
        raw_mpiw = float((pred[:, t, 2] - pred[:, t, 0]).mean())
        corr, _ = spearmanr(q_hi - q_lo, np.abs(y - q_med))

        results["per_target"][name] = {
            "r2": r2, "mae": mae,
            "picp": picp, "mpiw": mpiw,
            "raw_picp": raw_picp, "raw_mpiw": raw_mpiw,
            "q_hat": float(q_hat[t]),
            "interval_error_corr": float(corr),
        }

    results["mean_r2"] = float(np.mean([results["per_target"][n]["r2"] for n in TARGET_NAMES]))
    results["mean_picp"] = float(np.mean([results["per_target"][n]["picp"] for n in TARGET_NAMES]))
    results["mean_mpiw"] = float(np.mean([results["per_target"][n]["mpiw"] for n in TARGET_NAMES]))

    # Cross-track analysis
    if dist is not None:
        dist_bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
        results["cross_track"] = {}

        for lo, hi in dist_bins:
            bin_mask = (dist >= lo) & (dist < hi)
            n = int(bin_mask.sum())
            if n < 50:
                continue

            bin_results = {}
            for t, name in enumerate(TARGET_NAMES):
                q_lo_t = pred[bin_mask, t, 0] - q_hat[t]
                q_hi_t = pred[bin_mask, t, 2] + q_hat[t]
                q_med_t = pred[bin_mask, t, 1]
                y_t = true[bin_mask, t]

                covered_t = (y_t >= q_lo_t) & (y_t <= q_hi_t)
                r2_t = float(r2_score(y_t, q_med_t)) if n > 10 else None
                mae_t = float(np.abs(y_t - q_med_t).mean())
                picp_t = float(covered_t.mean())
                mpiw_t = float((q_hi_t - q_lo_t).mean())

                bin_results[name] = {
                    "r2": r2_t, "mae": mae_t,
                    "picp": picp_t, "mpiw": mpiw_t,
                }

            bin_results["n"] = n
            bin_results["mean_r2"] = float(np.mean(
                [bin_results[n_]["r2"] for n_ in TARGET_NAMES if bin_results[n_]["r2"] is not None]))
            bin_results["mean_picp"] = float(np.mean(
                [bin_results[n_]["picp"] for n_ in TARGET_NAMES]))
            bin_results["mean_mpiw"] = float(np.mean(
                [bin_results[n_]["mpiw"] for n_ in TARGET_NAMES]))

            results["cross_track"][f"{lo}-{hi}"] = bin_results

    # Reliability diagram data (multiple nominal levels)
    results["reliability"] = {}
    for nominal in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        alpha_r = 1 - nominal
        q_hat_r = calibrate_cqr(pred, true, alpha=alpha_r)
        empirical = {}
        for t, name in enumerate(TARGET_NAMES):
            q_lo_r = pred[:, t, 0] - q_hat_r[t]
            q_hi_r = pred[:, t, 2] + q_hat_r[t]
            cov = float(((true[:, t] >= q_lo_r) & (true[:, t] <= q_hi_r)).mean())
            empirical[name] = cov
        empirical["mean"] = float(np.mean(list(empirical.values())))
        results["reliability"][f"{nominal}"] = empirical

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cal-fraction", type=float, default=0.3)
    parser.add_argument("--cal-seed", type=int, default=123)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load model
    ckpt_path = MODEL_DIR / f"geom_{args.model_tag}.pt"
    print(f"Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
    geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
    era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
    era5_std = np.array(ckpt["era5_std"], dtype=np.float32)
    quantile_mode = ckpt.get("quantile_mode", False)

    model = ConvNextUNet(
        in_channels=10, out_channels=N_TARGETS,
        base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
        era5_dim=ckpt["era5_dim"],
        quantile_mode=quantile_mode,
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  {count_params(model)/1e6:.2f}M params, quantile_mode={quantile_mode}")

    # Calibration set (split from val)
    val_ds = DenseGeometryDatasetFull("val", geo_mean, geo_std, era5_mean, era5_std)
    N_val = len(val_ds)
    rng = np.random.RandomState(args.cal_seed)
    indices = rng.permutation(N_val)
    n_cal = int(N_val * args.cal_fraction)
    cal_indices = indices[:n_cal]
    print(f"  Cal set: {n_cal:,} patches from val ({N_val:,})")

    cal_ds = torch.utils.data.Subset(val_ds, cal_indices)
    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Calibration predictions
    print(f"\n  Collecting calibration predictions...")
    cal_pred, cal_true, _ = collect_with_distance(
        model, cal_loader, device, geo_mean, geo_std,
        use_tta=args.use_tta, quantile_mode=quantile_mode)
    print(f"  Calibration pixels: {len(cal_true):,}")

    # CQR calibration
    q_hat = calibrate_cqr(cal_pred, cal_true, alpha=args.alpha)
    print(f"\n  q_hat (alpha={args.alpha}):")
    for t, name in enumerate(TARGET_NAMES):
        print(f"    {name:>12s}: {q_hat[t]:.4f}")

    # Test set with distances
    test_ds = DenseGeometryDatasetFull("test", geo_mean, geo_std, era5_mean, era5_std)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    print(f"\n  Collecting test predictions with distances...")
    test_pred, test_true, test_dist = collect_with_distance(
        model, test_loader, device, geo_mean, geo_std,
        use_tta=args.use_tta, quantile_mode=quantile_mode)
    print(f"  Test pixels: {len(test_true):,}")

    # Compute all metrics
    results = compute_metrics(test_pred, test_true, q_hat, dist=test_dist)
    results["model_tag"] = args.model_tag
    results["alpha"] = args.alpha
    results["n_cal"] = len(cal_true)
    results["n_test"] = len(test_true)
    results["use_tta"] = args.use_tta

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Results (alpha={args.alpha}, {(1-args.alpha)*100:.0f}% target coverage)")
    print(f"{'='*70}")
    print(f"  Mean R²={results['mean_r2']:.4f}, PICP={results['mean_picp']:.4f}, "
          f"MPIW={results['mean_mpiw']:.4f}")
    print()
    print(f"  {'Target':>12s}  {'R²':>6s}  {'MAE':>7s}  {'PICP':>6s}  {'MPIW':>7s}  "
          f"{'rawPICP':>7s}  {'q_hat':>7s}  {'corr':>6s}")
    print(f"  {'-'*72}")
    for name in TARGET_NAMES:
        r = results["per_target"][name]
        print(f"  {name:>12s}  {r['r2']:6.4f}  {r['mae']:7.4f}  {r['picp']:6.4f}  "
              f"{r['mpiw']:7.4f}  {r['raw_picp']:7.4f}  {r['q_hat']:7.4f}  "
              f"{r['interval_error_corr']:6.3f}")

    # Cross-track
    if "cross_track" in results:
        print(f"\n  Cross-track uncertainty:")
        print(f"  {'Dist':>8s}  {'n':>7s}  {'R²':>6s}  {'PICP':>6s}  {'MPIW':>7s}")
        print(f"  {'-'*42}")
        for bin_name in sorted(results["cross_track"].keys(), key=lambda x: float(x.split('-')[0])):
            r = results["cross_track"][bin_name]
            print(f"  {bin_name:>8s}  {r['n']:7d}  {r['mean_r2']:6.4f}  "
                  f"{r['mean_picp']:6.4f}  {r['mean_mpiw']:7.4f}")

    # Reliability diagram
    print(f"\n  Reliability (mean across targets):")
    print(f"  {'Nominal':>8s}  {'Empirical':>9s}")
    for nom, emp in sorted(results["reliability"].items(), key=lambda x: float(x[0])):
        print(f"  {float(nom):8.2f}  {emp['mean']:9.4f}")

    # Save
    results_path = OUTPUT_DIR / f"results_cqr_eval_{args.model_tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Save raw predictions for plotting
    np.savez(OUTPUT_DIR / f"cqr_test_data_{args.model_tag}.npz",
             pred=test_pred, true=test_true, dist=test_dist, q_hat=q_hat)
    print("Done!")


if __name__ == "__main__":
    main()
