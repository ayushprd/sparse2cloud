"""Conformalized Quantile Regression (CQR) calibration.

Implements split conformal prediction (Romano et al., 2019) to produce
calibrated prediction intervals with guaranteed marginal coverage.

Procedure:
  1. Split val set into val_proper (model selection) and cal (calibration)
  2. Run inference on calibration set with trained quantile model
  3. Per target: compute nonconformity scores s = max(q_lo - y, y - q_hi)
  4. Find threshold q_hat at level ceil((n+1)(1-alpha))/n
  5. At test time: corrected interval = [q_lo - q_hat, q_hi + q_hat]

Usage:
    python -u 49_calibrate_cqr.py --model-tag G-QP --alphas 0.1,0.2,0.05
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast
from sklearn.metrics import r2_score
from pathlib import Path

from config import COLOC_DIR, OUTPUT_DIR, MODEL_DIR
from models.convnext_unet import ConvNextUNet, count_params

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


class DenseGeometryDataset(Dataset):
    """Minimal dataset for inference — loads geometry targets + ERA5."""

    def __init__(self, split, geo_mean, geo_std, era5_mean, era5_std):
        self.patches = np.load(DENSE_DIR / f"{split}_patches.npy")
        self.geometry = np.load(DENSE_DIR / f"{split}_geometry.npy")
        self.positions = np.load(DENSE_DIR / f"{split}_positions.npy")
        self.n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")
        self.n_samples = len(self.patches)
        self.geo_mean = geo_mean
        self.geo_std = geo_std

        # ERA5
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


def tta_predict(model, patch, era5, era5_valid):
    """TTA-8 for quantile model. Returns (B, T, Q, H, W)."""
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
                pred = torch.rot90(pred, -k, [3, 4])
            preds.append(pred)

    return torch.stack(preds).mean(0)


def collect_predictions(model, loader, device, geo_mean, geo_std, use_tta=False):
    """Run inference and collect per-pixel predictions + targets.

    Returns:
        pred_all: (N, 8, 3) quantile predictions in original units
        true_all: (N, 8) true values in original units
    """
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            if use_tta:
                pred = tta_predict(model, patch, era5, era5_valid)
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                pred = pred.float()

            pred = pred.cpu()  # (B, T, Q, H, W)
            target = target.float().cpu()
            mask = mask.cpu()

            B = pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                # pred: (T, Q) per pixel, denormalize
                for r, c in zip(rows, cols):
                    p = pred[b, :, :, r, c].numpy() * geo_std[:, None] + geo_mean[:, None]  # (8, 3)
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean  # (8,)
                    all_pred.append(p)
                    all_true.append(t)

    return np.stack(all_pred), np.stack(all_true)  # (N, 8, 3), (N, 8)


def calibrate_cqr(pred, true, alpha=0.1):
    """Compute CQR calibration thresholds per target.

    Args:
        pred: (N, 8, 3) quantile predictions [q_lo, q_med, q_hi]
        true: (N, 8) true values
        alpha: miscoverage level (e.g. 0.1 for 90% intervals)

    Returns:
        q_hat: (8,) per-target calibration thresholds
    """
    N = len(true)
    q_hat = np.zeros(N_TARGETS)

    for t in range(N_TARGETS):
        q_lo = pred[:, t, 0]
        q_hi = pred[:, t, 2]
        y = true[:, t]

        # Nonconformity scores
        scores = np.maximum(q_lo - y, y - q_hi)

        # Quantile at level ceil((n+1)(1-alpha)) / n
        level = np.ceil((N + 1) * (1 - alpha)) / N
        level = min(level, 1.0)
        q_hat[t] = np.quantile(scores, level)

    return q_hat


def evaluate_intervals(pred, true, q_hat):
    """Evaluate calibrated prediction intervals.

    Args:
        pred: (N, 8, 3) [q_lo, q_med, q_hi]
        true: (N, 8)
        q_hat: (8,) calibration thresholds

    Returns:
        dict with per-target PICP, MPIW, R², MAE
    """
    results = {}
    for t, name in enumerate(TARGET_NAMES):
        q_lo = pred[:, t, 0] - q_hat[t]
        q_hi = pred[:, t, 2] + q_hat[t]
        q_med = pred[:, t, 1]
        y = true[:, t]

        # PICP: Prediction Interval Coverage Probability
        covered = (y >= q_lo) & (y <= q_hi)
        picp = float(covered.mean())

        # MPIW: Mean Prediction Interval Width
        mpiw = float((q_hi - q_lo).mean())

        # Raw interval width (before CQR)
        raw_lo = pred[:, t, 0]
        raw_hi = pred[:, t, 2]
        raw_covered = (y >= raw_lo) & (y <= raw_hi)
        raw_picp = float(raw_covered.mean())
        raw_mpiw = float((raw_hi - raw_lo).mean())

        # Point estimate quality (median)
        r2 = float(r2_score(y, q_med))
        mae = float(np.abs(y - q_med).mean())

        # Interval-error correlation
        interval_width = q_hi - q_lo
        abs_error = np.abs(y - q_med)
        from scipy.stats import spearmanr
        corr, pval = spearmanr(interval_width, abs_error)

        results[name] = {
            "r2": r2,
            "mae": mae,
            "picp": picp,
            "mpiw": mpiw,
            "raw_picp": raw_picp,
            "raw_mpiw": raw_mpiw,
            "q_hat": float(q_hat[t]),
            "interval_error_corr": float(corr),
            "interval_error_pval": float(pval),
        }

    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["mean_picp"] = float(np.mean([results[n]["picp"] for n in TARGET_NAMES]))
    results["mean_mpiw"] = float(np.mean([results[n]["mpiw"] for n in TARGET_NAMES]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="0.1,0.2,0.05",
                        help="Comma-separated miscoverage levels")
    parser.add_argument("--cal-fraction", type=float, default=0.3,
                        help="Fraction of val set used for calibration")
    parser.add_argument("--cal-seed", type=int, default=123)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda")
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load model
    ckpt_path = MODEL_DIR / f"geom_{args.model_tag}.pt"
    print(f"Loading model: {ckpt_path}")
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
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  {count_params(model)/1e6:.2f}M params")

    # Split val set into val_proper and calibration
    val_ds = DenseGeometryDataset("val", geo_mean, geo_std, era5_mean, era5_std)
    N_val = len(val_ds)
    rng = np.random.RandomState(args.cal_seed)
    indices = rng.permutation(N_val)
    n_cal = int(N_val * args.cal_fraction)
    cal_indices = indices[:n_cal]
    val_proper_indices = indices[n_cal:]
    print(f"\n  Val set: {N_val:,} total → {len(val_proper_indices):,} val + {n_cal:,} cal")

    cal_ds = Subset(val_ds, cal_indices)
    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Collect calibration predictions
    print(f"\n  Running inference on calibration set (TTA={args.use_tta})...")
    cal_pred, cal_true = collect_predictions(
        model, cal_loader, device, geo_mean, geo_std, use_tta=args.use_tta)
    print(f"  Calibration pixels: {len(cal_true):,}")

    # Collect test predictions
    test_ds = DenseGeometryDataset("test", geo_mean, geo_std, era5_mean, era5_std)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(f"\n  Running inference on test set (TTA={args.use_tta})...")
    test_pred, test_true = collect_predictions(
        model, test_loader, device, geo_mean, geo_std, use_tta=args.use_tta)
    print(f"  Test pixels: {len(test_true):,}")

    # Calibrate and evaluate at each alpha
    all_results = {}
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"  CQR calibration at alpha={alpha} ({(1-alpha)*100:.0f}% coverage)")
        print(f"{'='*60}")

        q_hat = calibrate_cqr(cal_pred, cal_true, alpha=alpha)
        results = evaluate_intervals(test_pred, test_true, q_hat)

        print(f"\n  Mean R² (median): {results['mean_r2']:.4f}")
        print(f"  Mean PICP: {results['mean_picp']:.4f} (target: {1-alpha:.2f})")
        print(f"  Mean MPIW: {results['mean_mpiw']:.4f}")
        print()
        print(f"  {'Target':>12s}  {'R²':>6s}  {'MAE':>7s}  {'PICP':>6s}  "
              f"{'MPIW':>7s}  {'rawPICP':>7s}  {'rawMPIW':>7s}  {'q_hat':>7s}  {'corr':>6s}")
        print(f"  {'-'*80}")
        for name in TARGET_NAMES:
            r = results[name]
            print(f"  {name:>12s}  {r['r2']:6.4f}  {r['mae']:7.4f}  {r['picp']:6.4f}  "
                  f"{r['mpiw']:7.4f}  {r['raw_picp']:7.4f}  {r['raw_mpiw']:7.4f}  "
                  f"{r['q_hat']:7.4f}  {r['interval_error_corr']:6.3f}")

        all_results[f"alpha_{alpha}"] = results

    # Save predictions for downstream analysis
    save_data = {
        "cal_pred": cal_pred,  # (N_cal, 8, 3)
        "cal_true": cal_true,  # (N_cal, 8)
        "test_pred": test_pred,  # (N_test, 8, 3)
        "test_true": test_true,  # (N_test, 8)
    }
    np.savez(OUTPUT_DIR / f"cqr_predictions_{args.model_tag}.npz", **save_data)

    # Save q_hat per alpha
    for alpha in alphas:
        q_hat = calibrate_cqr(cal_pred, cal_true, alpha=alpha)
        all_results[f"alpha_{alpha}"]["q_hat_per_target"] = {
            name: float(q_hat[i]) for i, name in enumerate(TARGET_NAMES)
        }

    all_results["model_tag"] = args.model_tag
    all_results["cal_fraction"] = args.cal_fraction
    all_results["n_cal_pixels"] = len(cal_true)
    all_results["n_test_pixels"] = len(test_true)
    all_results["use_tta"] = args.use_tta

    results_path = OUTPUT_DIR / f"results_cqr_{args.model_tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results: {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
