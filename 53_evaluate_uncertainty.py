"""Unified uncertainty evaluation: all methods → CQR calibration → identical metrics.

Loads any model type, extracts prediction intervals, applies CQR calibration,
and computes identical metrics for fair comparison.

Supported modes:
  - quantile: standard quantile regression (G-Q, G-QP)
  - physics_quantile: physics-constrained quantile (G-QPC)
  - quantile_ensemble: ensemble of quantile models (G-QE)
  - point: point prediction only (G-B, G-PC) — intervals from residual bootstrap
  - heteroscedastic: beta-NLL (G-HET)
  - evidential: NIG regression (G-EVI)
  - mc_dropout: MC-Dropout from point model (G-MCD)
  - ensemble: deep ensemble of point models (G-DE)

Usage:
    # Quantile models
    python -u 53_evaluate_uncertainty.py --model-tag G-QP --mode quantile

    # Physics-constrained quantile
    python -u 53_evaluate_uncertainty.py --model-tag G-QPC --mode physics_quantile

    # Heteroscedastic
    python -u 53_evaluate_uncertainty.py --model-tag G-HET --mode heteroscedastic

    # Evidential
    python -u 53_evaluate_uncertainty.py --model-tag G-EVI --mode evidential

    # MC-Dropout (uses G-B checkpoint)
    python -u 53_evaluate_uncertainty.py --model-tag G-MCD --mode mc_dropout --base-model geom_G-B.pt --T 30

    # Deep Ensemble
    python -u 53_evaluate_uncertainty.py --model-tag G-DE --mode ensemble --ensemble-tags G-B,G-DE2,G-DE3,G-DE4,G-DE5

    # Quantile Ensemble
    python -u 53_evaluate_uncertainty.py --model-tag G-QE --mode quantile_ensemble --ensemble-tags G-Q,G-Q2,G-Q3,G-Q4,G-Q5

    # Compare all methods
    python -u 53_evaluate_uncertainty.py --compare
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt
from pathlib import Path

from config import COLOC_DIR, OUTPUT_DIR, MODEL_DIR, FIGURE_DIR
from models.convnext_unet import ConvNextUNet, count_params

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


class DenseGeometryDataset(Dataset):
    """Dataset for inference with normalized geometry + ERA5."""

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


def load_model(ckpt_path, mode, device):
    """Load a model from checkpoint.

    Returns: model, geo_mean, geo_std, era5_mean, era5_std
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
    geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
    era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
    era5_std = np.array(ckpt["era5_std"], dtype=np.float32)

    # Determine output config from mode
    if mode in ("quantile", "physics_quantile"):
        quantile_mode = True
        physics_head = (mode == "physics_quantile")
        out_channels = N_TARGETS
    elif mode == "heteroscedastic":
        quantile_mode = False
        physics_head = False
        out_channels = N_TARGETS * 2
    elif mode == "evidential":
        quantile_mode = False
        physics_head = False
        out_channels = N_TARGETS * 4
    else:  # point, mc_dropout, ensemble member
        quantile_mode = False
        physics_head = ckpt.get("physics_head", False)
        out_channels = N_TARGETS

    model = ConvNextUNet(
        in_channels=10, out_channels=out_channels,
        base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
        era5_dim=ckpt["era5_dim"],
        quantile_mode=quantile_mode,
        physics_head=physics_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, geo_mean, geo_std, era5_mean, era5_std


def collect_intervals_quantile(model, loader, device, geo_mean, geo_std, use_tta=False):
    """Collect intervals from quantile/physics-quantile model.

    Returns: pred (N,8,3), true (N,8), dist (N,)
    """
    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            if use_tta:
                preds = []
                for k in range(4):
                    for flip in [False, True]:
                        x = torch.rot90(patch, k, [2, 3])
                        if flip:
                            x = x.flip(-1)
                        with autocast("cuda", dtype=torch.bfloat16):
                            p = model(x, era5, era5_valid)
                        p = p.float()
                        if flip:
                            p = p.flip(-1)
                        if k > 0:
                            p = torch.rot90(p, -k, [3, 4])
                        preds.append(p)
                pred = torch.stack(preds).mean(0)
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                pred = pred.float()

            pred = pred.cpu()  # (B, 8, 3, H, W)
            target = target.float().cpu()
            mask = mask.cpu()

            B = pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np)

                for r, c in zip(rows, cols):
                    p = pred[b, :, :, r, c].numpy() * geo_std[:, None] + geo_mean[:, None]
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


def collect_intervals_parametric(model, loader, device, geo_mean, geo_std, mode):
    """Collect intervals from heteroscedastic/evidential model.

    Returns: pred (N,8,3), true (N,8), dist (N,)
    """
    from models.evidential_loss import parse_nig
    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(patch, era5, era5_valid)
            out = out.float().cpu()
            target = target.float().cpu()
            mask = mask.cpu()

            B = out.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np)

                for r, c in zip(rows, cols):
                    if mode == "heteroscedastic":
                        mean_n = out[b, :N_TARGETS, r, c].numpy()
                        log_var_n = out[b, N_TARGETS:, r, c].numpy().clip(-10, 10)
                        std_n = np.exp(0.5 * log_var_n)
                        # Denormalize
                        mean = mean_n * geo_std + geo_mean
                        std = std_n * geo_std
                        lo = mean - 1.28 * std
                        hi = mean + 1.28 * std
                    elif mode == "evidential":
                        raw = out[b, :, r, c].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, 4T, 1, 1)
                        gamma, nu, alpha, beta = parse_nig(raw, N_TARGETS)
                        gamma = gamma[0, :, 0, 0].numpy()
                        alpha_v = alpha[0, :, 0, 0].numpy()
                        beta_v = beta[0, :, 0, 0].numpy()
                        total_var = beta_v / np.maximum(alpha_v - 1, 0.01)
                        std_n = np.sqrt(total_var)
                        mean = gamma * geo_std + geo_mean
                        std = std_n * geo_std
                        lo = mean - 1.28 * std
                        hi = mean + 1.28 * std

                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    p = np.stack([lo, mean, hi], axis=1)  # (8, 3)
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


def collect_intervals_mc_dropout(model_path, loader, device, geo_mean, geo_std, T=30):
    """MC-Dropout: T forward passes with dropout enabled.

    Returns: pred (N,8,3), true (N,8), dist (N,)
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ConvNextUNet(
        in_channels=10, out_channels=N_TARGETS,
        base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
        era5_dim=ckpt["era5_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Enable only Dropout2d at test time
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()

    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            preds = []
            for _ in range(T):
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                preds.append(pred.float().cpu())

            preds = torch.stack(preds)  # (T, B, 8, H, W)
            mean = preds.mean(0)
            std = preds.std(0)

            target = target.float().cpu()
            mask = mask.cpu()

            B = mean.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np)

                for r, c in zip(rows, cols):
                    m = mean[b, :, r, c].numpy() * geo_std + geo_mean
                    s = std[b, :, r, c].numpy() * geo_std
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    p = np.stack([m - 1.28 * s, m, m + 1.28 * s], axis=1)
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

            if (bi + 1) % 50 == 0:
                n_total = sum(len(p) for p in all_pred)
                print(f"    [{bi+1}/{len(loader)}] {n_total:,} pixels")

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


def collect_intervals_ensemble(model_paths, loader, device, geo_mean, geo_std):
    """Deep Ensemble: average over M models.

    Returns: pred (N,8,3), true (N,8), dist (N,)
    """
    models = []
    for path in model_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ConvNextUNet(
            in_channels=10, out_channels=N_TARGETS,
            base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
            era5_dim=ckpt["era5_dim"],
        ).to(device).eval()
        m.load_state_dict(ckpt["model_state_dict"])
        models.append(m)
    print(f"  Loaded {len(models)} ensemble members")

    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            preds = []
            for m in models:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = m(patch, era5, era5_valid)
                preds.append(pred.float().cpu())

            preds = torch.stack(preds)  # (M, B, 8, H, W)
            mean = preds.mean(0)
            std = preds.std(0)

            target = target.float().cpu()
            mask = mask.cpu()

            B = mean.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np)

                for r, c in zip(rows, cols):
                    m_val = mean[b, :, r, c].numpy() * geo_std + geo_mean
                    s = std[b, :, r, c].numpy() * geo_std
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    p = np.stack([m_val - 1.28 * s, m_val, m_val + 1.28 * s], axis=1)
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

            if (bi + 1) % 50 == 0:
                n_total = sum(len(p) for p in all_pred)
                print(f"    [{bi+1}/{len(loader)}] {n_total:,} pixels")

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


def collect_intervals_quantile_ensemble(model_paths, loader, device, geo_mean, geo_std):
    """Quantile Ensemble: average quantile predictions from M quantile models.

    For each pixel:
      - median = mean of M median predictions
      - q_lo = mean of M q_lo predictions
      - q_hi = mean of M q_hi predictions

    Returns: pred (N,8,3), true (N,8), dist (N,)
    """
    models = []
    for path in model_paths:
        m, *_ = load_model(path, "quantile", device)
        models.append(m)
    print(f"  Loaded {len(models)} quantile ensemble members")

    all_pred, all_true, all_dist = [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            # Get predictions from all models
            member_preds = []
            for m in models:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = m(patch, era5, era5_valid)
                member_preds.append(pred.float().cpu())

            # Stack and average: (M, B, 8, 3, H, W) → (B, 8, 3, H, W)
            ensemble_pred = torch.stack(member_preds).mean(0)

            target = target.float().cpu()
            mask = mask.cpu()

            B = ensemble_pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                mask_np = mask[b].numpy()
                dist_map = distance_transform_edt(1 - mask_np)

                for r, c in zip(rows, cols):
                    p = ensemble_pred[b, :, :, r, c].numpy() * geo_std[:, None] + geo_mean[:, None]
                    t = target[b, :, r, c].numpy() * geo_std + geo_mean
                    all_pred.append(p)
                    all_true.append(t)
                    all_dist.append(dist_map[r.item(), c.item()])

            if (bi + 1) % 50 == 0:
                n_total = sum(len(p) for p in all_pred)
                print(f"    [{bi+1}/{len(loader)}] {n_total:,} pixels")

    return np.stack(all_pred), np.stack(all_true), np.array(all_dist)


# CQR Calibration
def calibrate_cqr(pred, true, alpha=0.1):
    """CQR calibration. Returns per-target q_hat."""
    N = len(true)
    q_hat = np.zeros(N_TARGETS)
    for t in range(N_TARGETS):
        scores = np.maximum(pred[:, t, 0] - true[:, t], true[:, t] - pred[:, t, 2])
        level = min(np.ceil((N + 1) * (1 - alpha)) / N, 1.0)
        q_hat[t] = np.quantile(scores, level)
    return q_hat


# Metrics
def compute_metrics(pred, true, q_hat, dist=None, alpha=0.1):
    """Comprehensive metrics identical across all methods."""
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
    results["mean_mae"] = float(np.mean([results["per_target"][n]["mae"] for n in TARGET_NAMES]))
    results["mean_picp"] = float(np.mean([results["per_target"][n]["picp"] for n in TARGET_NAMES]))
    results["mean_mpiw"] = float(np.mean([results["per_target"][n]["mpiw"] for n in TARGET_NAMES]))
    results["mean_raw_picp"] = float(np.mean([results["per_target"][n]["raw_picp"] for n in TARGET_NAMES]))
    results["mean_raw_mpiw"] = float(np.mean([results["per_target"][n]["raw_mpiw"] for n in TARGET_NAMES]))
    results["mean_corr"] = float(np.mean([results["per_target"][n]["interval_error_corr"] for n in TARGET_NAMES]))

    # Cross-track analysis
    if dist is not None:
        dist_bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
        results["cross_track"] = {}
        for lo, hi in dist_bins:
            bin_mask = (dist >= lo) & (dist < hi)
            n = int(bin_mask.sum())
            if n < 50:
                continue
            bin_res = {}
            for t2, name2 in enumerate(TARGET_NAMES):
                q_lo2 = pred[bin_mask, t2, 0] - q_hat[t2]
                q_hi2 = pred[bin_mask, t2, 2] + q_hat[t2]
                q_med2 = pred[bin_mask, t2, 1]
                y2 = true[bin_mask, t2]
                cov = (y2 >= q_lo2) & (y2 <= q_hi2)
                bin_res[name2] = {
                    "r2": float(r2_score(y2, q_med2)) if n > 10 else None,
                    "picp": float(cov.mean()),
                    "mpiw": float((q_hi2 - q_lo2).mean()),
                }
            bin_res["n"] = n
            bin_res["mean_r2"] = float(np.mean([
                bin_res[n2]["r2"] for n2 in TARGET_NAMES if bin_res[n2]["r2"] is not None]))
            bin_res["mean_picp"] = float(np.mean([bin_res[n2]["picp"] for n2 in TARGET_NAMES]))
            results["cross_track"][f"{lo}-{hi}"] = bin_res

    # Reliability diagram (multiple nominal levels)
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
        empirical["mean"] = float(np.mean([empirical[n] for n in TARGET_NAMES]))
        results["reliability"][f"{nominal}"] = empirical

    return results


def print_results(results, tag, mode):
    """Pretty-print evaluation results."""
    print(f"\n{'='*75}")
    print(f"  {tag} ({mode})")
    print(f"  Mean R²={results['mean_r2']:.4f}, PICP={results['mean_picp']:.4f}, "
          f"MPIW={results['mean_mpiw']:.4f}, corr={results['mean_corr']:.3f}")
    print(f"{'='*75}")
    print(f"  {'Target':>12s}  {'R²':>6s}  {'MAE':>7s}  {'PICP':>6s}  {'MPIW':>7s}  "
          f"{'rawPICP':>7s}  {'rawMPIW':>7s}  {'q_hat':>7s}  {'corr':>6s}")
    print(f"  {'-'*75}")
    for name in TARGET_NAMES:
        r = results["per_target"][name]
        print(f"  {name:>12s}  {r['r2']:6.4f}  {r['mae']:7.4f}  {r['picp']:6.4f}  "
              f"{r['mpiw']:7.4f}  {r['raw_picp']:7.4f}  {r['raw_mpiw']:7.4f}  "
              f"{r['q_hat']:7.4f}  {r['interval_error_corr']:6.3f}")


def compare_all():
    """Load all saved evaluation results and produce comparison table."""
    import glob

    result_files = sorted(glob.glob(str(OUTPUT_DIR / "results_uncertainty_*.json")))
    if not result_files:
        print("No evaluation results found. Run individual evaluations first.")
        return

    all_results = {}
    for f in result_files:
        with open(f) as fp:
            data = json.load(fp)
        tag = data["model_tag"]
        all_results[tag] = data

    # Print comparison table
    tags = sorted(all_results.keys())
    print(f"\n{'='*100}")
    print(f"  COMPARISON TABLE — All Methods (CQR-calibrated)")
    print(f"{'='*100}")

    # Summary row
    print(f"\n  {'Method':>12s}  {'Mode':>18s}  {'R²':>6s}  {'MAE':>7s}  {'PICP':>6s}  "
          f"{'MPIW':>7s}  {'rawPICP':>7s}  {'corr':>6s}")
    print(f"  {'-'*80}")
    for tag in tags:
        r = all_results[tag]
        print(f"  {tag:>12s}  {r['mode']:>18s}  {r['mean_r2']:6.4f}  {r['mean_mae']:7.4f}  "
              f"{r['mean_picp']:6.4f}  {r['mean_mpiw']:7.4f}  {r['mean_raw_picp']:7.4f}  "
              f"{r['mean_corr']:6.3f}")

    # Per-target R²
    print(f"\n  Per-target R²:")
    print(f"  {'Method':>12s}", end="")
    for name in TARGET_NAMES:
        print(f"  {name:>10s}", end="")
    print(f"  {'Mean':>8s}")
    print(f"  {'-'*110}")
    for tag in tags:
        r = all_results[tag]
        print(f"  {tag:>12s}", end="")
        for name in TARGET_NAMES:
            print(f"  {r['per_target'][name]['r2']:10.4f}", end="")
        print(f"  {r['mean_r2']:8.4f}")

    # Per-target PICP
    print(f"\n  Per-target PICP (CQR, target ≥ 0.90):")
    print(f"  {'Method':>12s}", end="")
    for name in TARGET_NAMES:
        print(f"  {name:>10s}", end="")
    print(f"  {'Mean':>8s}")
    print(f"  {'-'*110}")
    for tag in tags:
        r = all_results[tag]
        print(f"  {tag:>12s}", end="")
        for name in TARGET_NAMES:
            print(f"  {r['per_target'][name]['picp']:10.4f}", end="")
        print(f"  {r['mean_picp']:8.4f}")

    # Per-target MPIW (lower is better, given PICP ≥ nominal)
    print(f"\n  Per-target MPIW (lower = sharper, given PICP ≥ nominal):")
    print(f"  {'Method':>12s}", end="")
    for name in TARGET_NAMES:
        print(f"  {name:>10s}", end="")
    print(f"  {'Mean':>8s}")
    print(f"  {'-'*110}")
    for tag in tags:
        r = all_results[tag]
        print(f"  {tag:>12s}", end="")
        for name in TARGET_NAMES:
            print(f"  {r['per_target'][name]['mpiw']:10.4f}", end="")
        print(f"  {r['mean_mpiw']:8.4f}")

    # Interval-error correlation (higher = better uncertainty ranking)
    print(f"\n  Interval-Error Spearman ρ (higher = better uncertainty ranking):")
    print(f"  {'Method':>12s}", end="")
    for name in TARGET_NAMES:
        print(f"  {name:>10s}", end="")
    print(f"  {'Mean':>8s}")
    print(f"  {'-'*110}")
    for tag in tags:
        r = all_results[tag]
        print(f"  {tag:>12s}", end="")
        for name in TARGET_NAMES:
            print(f"  {r['per_target'][name]['interval_error_corr']:10.3f}", end="")
        print(f"  {r['mean_corr']:8.3f}")

    # Save as JSON
    comparison_path = OUTPUT_DIR / "results_uncertainty_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Comparison saved: {comparison_path}")


def evaluate_single(args):
    """Evaluate a single method."""
    device = torch.device("cuda")
    mode = args.mode
    tag = args.model_tag
    alpha = args.alpha
    t0 = time.time()

    print(f"\nEvaluating {tag} (mode={mode}, alpha={alpha})")

    # Load normalization stats from appropriate checkpoint
    if mode in ("mc_dropout", "ensemble"):
        base_ckpt = torch.load(MODEL_DIR / args.base_model, map_location="cpu", weights_only=False)
    elif mode == "quantile_ensemble":
        # Use first ensemble member's stats
        first_tag = args.ensemble_tags.split(",")[0]
        base_ckpt = torch.load(MODEL_DIR / f"geom_{first_tag}.pt", map_location="cpu", weights_only=False)
    else:
        base_ckpt = torch.load(MODEL_DIR / f"geom_{tag}.pt", map_location="cpu", weights_only=False)

    geo_mean = np.array(base_ckpt["geo_mean"], dtype=np.float32)
    geo_std = np.array(base_ckpt["geo_std"], dtype=np.float32)
    era5_mean = np.array(base_ckpt["era5_mean"], dtype=np.float32)
    era5_std = np.array(base_ckpt["era5_std"], dtype=np.float32)

    # Setup datasets
    val_ds = DenseGeometryDataset("val", geo_mean, geo_std, era5_mean, era5_std)
    test_ds = DenseGeometryDataset("test", geo_mean, geo_std, era5_mean, era5_std)

    N_val = len(val_ds)
    rng = np.random.RandomState(args.cal_seed)
    indices = rng.permutation(N_val)
    n_cal = int(N_val * args.cal_fraction)
    cal_indices = indices[:n_cal]

    cal_ds = Subset(val_ds, cal_indices)

    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    print(f"  Cal: {n_cal:,} patches, Test: {len(test_ds):,} patches")

    # Collect predictions based on mode
    if mode in ("quantile", "physics_quantile"):
        model, *_ = load_model(MODEL_DIR / f"geom_{tag}.pt", mode, device)
        print(f"  Model: {count_params(model)/1e6:.2f}M params")
        print("  Collecting calibration predictions...")
        cal_pred, cal_true, _ = collect_intervals_quantile(
            model, cal_loader, device, geo_mean, geo_std, use_tta=args.use_tta)
        print(f"  Cal pixels: {len(cal_true):,}")
        print("  Collecting test predictions...")
        test_pred, test_true, test_dist = collect_intervals_quantile(
            model, test_loader, device, geo_mean, geo_std, use_tta=args.use_tta)

    elif mode in ("heteroscedastic", "evidential"):
        model, *_ = load_model(MODEL_DIR / f"geom_{tag}.pt", mode, device)
        print(f"  Model: {count_params(model)/1e6:.2f}M params")
        print("  Collecting calibration predictions...")
        cal_pred, cal_true, _ = collect_intervals_parametric(
            model, cal_loader, device, geo_mean, geo_std, mode)
        print(f"  Cal pixels: {len(cal_true):,}")
        print("  Collecting test predictions...")
        test_pred, test_true, test_dist = collect_intervals_parametric(
            model, test_loader, device, geo_mean, geo_std, mode)

    elif mode == "mc_dropout":
        print(f"  MC-Dropout T={args.T} from {args.base_model}")
        print("  Collecting calibration predictions...")
        cal_pred, cal_true, _ = collect_intervals_mc_dropout(
            MODEL_DIR / args.base_model, cal_loader, device, geo_mean, geo_std, T=args.T)
        print(f"  Cal pixels: {len(cal_true):,}")
        print("  Collecting test predictions...")
        test_pred, test_true, test_dist = collect_intervals_mc_dropout(
            MODEL_DIR / args.base_model, test_loader, device, geo_mean, geo_std, T=args.T)

    elif mode == "ensemble":
        ensemble_tags = args.ensemble_tags.split(",")
        model_paths = [MODEL_DIR / f"geom_{t}.pt" for t in ensemble_tags]
        for p in model_paths:
            if not p.exists():
                print(f"  WARNING: {p} does not exist, skipping")
        model_paths = [p for p in model_paths if p.exists()]
        print("  Collecting calibration predictions...")
        cal_pred, cal_true, _ = collect_intervals_ensemble(
            model_paths, cal_loader, device, geo_mean, geo_std)
        print(f"  Cal pixels: {len(cal_true):,}")
        print("  Collecting test predictions...")
        test_pred, test_true, test_dist = collect_intervals_ensemble(
            model_paths, test_loader, device, geo_mean, geo_std)

    elif mode == "quantile_ensemble":
        ensemble_tags = args.ensemble_tags.split(",")
        model_paths = [MODEL_DIR / f"geom_{t}.pt" for t in ensemble_tags]
        for p in model_paths:
            if not p.exists():
                print(f"  WARNING: {p} does not exist, skipping")
        model_paths = [p for p in model_paths if p.exists()]
        print(f"  Quantile ensemble: {len(model_paths)} members")
        print("  Collecting calibration predictions...")
        cal_pred, cal_true, _ = collect_intervals_quantile_ensemble(
            model_paths, cal_loader, device, geo_mean, geo_std)
        print(f"  Cal pixels: {len(cal_true):,}")
        print("  Collecting test predictions...")
        test_pred, test_true, test_dist = collect_intervals_quantile_ensemble(
            model_paths, test_loader, device, geo_mean, geo_std)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  Test pixels: {len(test_true):,}")

    # CQR calibration
    print(f"\n  CQR calibration (alpha={alpha})...")
    q_hat = calibrate_cqr(cal_pred, cal_true, alpha=alpha)
    for t, name in enumerate(TARGET_NAMES):
        print(f"    {name:>12s}: q_hat={q_hat[t]:.4f}")

    # Compute metrics
    results = compute_metrics(test_pred, test_true, q_hat, dist=test_dist, alpha=alpha)
    results["model_tag"] = tag
    results["mode"] = mode
    results["alpha"] = alpha
    results["n_cal"] = len(cal_true)
    results["n_test"] = len(test_true)
    results["use_tta"] = args.use_tta
    results["elapsed_s"] = time.time() - t0

    print_results(results, tag, mode)

    # Save
    results_path = OUTPUT_DIR / f"results_uncertainty_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Save predictions for downstream tasks
    npz_path = OUTPUT_DIR / f"cqr_test_data_{tag}.npz"
    np.savez(npz_path, pred=test_pred, true=test_true, dist=test_dist, q_hat=q_hat)
    print(f"  Predictions: {npz_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--mode", choices=[
        "quantile", "physics_quantile", "quantile_ensemble", "point",
        "heteroscedastic", "evidential", "mc_dropout", "ensemble"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cal-fraction", type=float, default=0.3)
    parser.add_argument("--cal-seed", type=int, default=123)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    # MC-Dropout
    parser.add_argument("--base-model", type=str, default="geom_G-B.pt")
    parser.add_argument("--T", type=int, default=30)
    # Ensemble
    parser.add_argument("--ensemble-tags", type=str, default="")
    # Compare mode
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_all()
    elif args.model_tag and args.mode:
        evaluate_single(args)
    else:
        parser.error("Provide --model-tag and --mode, or --compare")


if __name__ == "__main__":
    main()
