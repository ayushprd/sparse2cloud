"""Per-pixel MLP baseline for 8 geometry targets.

Demonstrates that the ConvNextUNet's spatial context improves over
a per-pixel MLP, justifying the architecture choice.

Also runs label efficiency sweep to show robustness to sparse supervision.

Usage:
    python -u 64_mlp_geometry_baseline.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from config import COLOC_DIR, OUTPUT_DIR

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


class PixelGeometryMLP(nn.Module):
    def __init__(self, input_dim, n_targets=8, hidden_dim=512, n_layers=5, dropout=0.3):
        super().__init__()
        layers = []
        dim = input_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = hidden_dim
        layers.append(nn.Linear(dim, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_geometry_pixels(split, context_size=5):
    """Extract supervised pixels with local context + ERA5 for geometry targets."""
    patches = np.load(DENSE_DIR / f"{split}_patches.npy")
    geometry = np.load(DENSE_DIR / f"{split}_geometry.npy")  # (N, 64, 8)
    positions = np.load(DENSE_DIR / f"{split}_positions.npy")
    n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")

    # Geometry normalization
    stats = np.load(OUTPUT_DIR / "geometry_stats.npz")
    geo_mean = stats["mean"].astype(np.float32)
    geo_std = stats["std"].astype(np.float32)

    # ERA5
    era5_raw = np.load(DENSE_DIR / f"{split}_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    era5_valid = (~nan_mask[:, 0, 0])
    era5_flat = era5_raw.reshape(len(era5_raw), -1)

    half = context_size // 2
    all_features, all_targets = [], []

    for i in range(len(patches)):
        n = int(n_profiles[i])
        if n == 0:
            continue
        rows = positions[i, :n, 0].astype(np.int64)
        cols = positions[i, :n, 1].astype(np.int64)
        valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
        rows, cols = rows[valid], cols[valid]
        if len(rows) == 0:
            continue

        # Per-pixel VIIRS channels
        pixel_features = patches[i][:, rows, cols].T  # (n_valid, 10)

        # Local context stats (5x5 region)
        if context_size > 1:
            patch_padded = np.pad(patches[i],
                                  ((0, 0), (half, half), (half, half)),
                                  mode='reflect')
            local_stats = []
            for r, c in zip(rows, cols):
                rp, cp = r + half, c + half
                region = patch_padded[:, rp-half:rp+half+1, cp-half:cp+half+1]
                local_stats.append(np.concatenate([
                    region.mean(axis=(1, 2)),
                    region.std(axis=(1, 2)),
                    region.min(axis=(1, 2)),
                    region.max(axis=(1, 2)),
                ]))
            local_stats = np.stack(local_stats)
            pixel_features = np.concatenate([pixel_features, local_stats], axis=1)

        # ERA5
        if era5_valid[i]:
            era5_tile = np.tile(era5_flat[i], (len(rows), 1))
        else:
            era5_tile = np.zeros((len(rows), era5_flat.shape[1]), dtype=np.float32)
        pixel_features = np.concatenate([pixel_features, era5_tile], axis=1)

        # Geometry targets (z-score normalized)
        geo = geometry[i, :n][valid]  # (n_valid, 8)
        geo_norm = (geo - geo_mean) / geo_std
        all_features.append(pixel_features)
        all_targets.append(geo_norm)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    print(f"  {split}: {len(features):,} pixels, dim={features.shape[1]}")
    return features, targets, geo_mean, geo_std


def train_and_evaluate(train_feat, train_tgt, val_feat, val_tgt,
                       test_feat, test_tgt, geo_mean, geo_std,
                       tag, device, epochs=200, patience=25,
                       lr=1e-3, batch_size=4096):
    """Train MLP and evaluate on test set."""
    input_dim = train_feat.shape[1]
    model = PixelGeometryMLP(input_dim, N_TARGETS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{tag}] MLP: {n_params/1e3:.1f}K params, input_dim={input_dim}, "
          f"train={len(train_feat):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    train_ds = TensorDataset(torch.from_numpy(train_feat), torch.from_numpy(train_tgt))
    val_ds = TensorDataset(torch.from_numpy(val_feat), torch.from_numpy(val_tgt))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.l1_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum, val_n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss_sum += F.l1_loss(model(x), y).item()
                val_n += 1
        val_loss = val_loss_sum / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1
            marker = f" (wait={wait})" if wait > 3 else ""

        if epoch % 20 == 0 or wait == 0 or wait >= patience:
            print(f"    ep {epoch:3d}: loss={loss_sum/n_batches:.5f} val={val_loss:.5f} "
                  f"[{time.time()-t0:.1f}s]{marker}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Evaluate best model
    model.load_state_dict(best_state)
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(test_feat))
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    all_pred = []
    with torch.no_grad():
        for (x,) in test_loader:
            all_pred.append(model(x.to(device)).cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)

    # Denormalize
    pred_denorm = pred * geo_std + geo_mean
    true_denorm = test_tgt * geo_std + geo_mean

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = float(r2_score(true_denorm[:, i], pred_denorm[:, i]))
        mae = float(np.abs(true_denorm[:, i] - pred_denorm[:, i]).mean())
        results[name] = {"r2": r2, "mae": mae}
    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["n_pixels"] = len(pred)
    results["n_params"] = n_params
    results["n_train"] = len(train_feat)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print("Extracting geometry pixels...")
    train_feat, train_tgt, geo_mean, geo_std = extract_geometry_pixels("train")
    val_feat, val_tgt, _, _ = extract_geometry_pixels("val")
    test_feat, test_tgt, _, _ = extract_geometry_pixels("test")

    # Normalize features (zero mean, unit variance from training set)
    feat_mean = train_feat.mean(axis=0)
    feat_std = train_feat.std(axis=0) + 1e-8
    train_feat = (train_feat - feat_mean) / feat_std
    val_feat = (val_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std

    all_results = {}

    # Experiment 1: Full MLP baseline
    print("\n" + "=" * 70)
    print("Experiment 1: Full MLP (100% labels)")
    print("=" * 70)
    results = train_and_evaluate(
        train_feat, train_tgt, val_feat, val_tgt,
        test_feat, test_tgt, geo_mean, geo_std,
        tag="MLP-100pct", device=device,
    )
    all_results["MLP-100pct"] = results
    print(f"\n  Mean R²: {results['mean_r2']:.4f}")
    for name in TARGET_NAMES:
        r = results[name]
        print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # Experiment 2: Per-pixel MLP without local context
    print("\n" + "=" * 70)
    print("Experiment 2: Per-pixel MLP (no local context, no ERA5)")
    print("=" * 70)
    # Re-extract with context_size=1 (pixel only, 10 channels)
    train_feat_px, train_tgt_px, _, _ = extract_geometry_pixels("train", context_size=1)
    val_feat_px, val_tgt_px, _, _ = extract_geometry_pixels("val", context_size=1)
    test_feat_px, test_tgt_px, _, _ = extract_geometry_pixels("test", context_size=1)

    # Only VIIRS channels (first 10 dims) — no ERA5
    n_viirs = 10
    train_feat_px_only = train_feat_px[:, :n_viirs]
    val_feat_px_only = val_feat_px[:, :n_viirs]
    test_feat_px_only = test_feat_px[:, :n_viirs]

    # Normalize
    px_mean = train_feat_px_only.mean(axis=0)
    px_std = train_feat_px_only.std(axis=0) + 1e-8
    train_feat_px_norm = (train_feat_px_only - px_mean) / px_std
    val_feat_px_norm = (val_feat_px_only - px_mean) / px_std
    test_feat_px_norm = (test_feat_px_only - px_mean) / px_std

    results_px = train_and_evaluate(
        train_feat_px_norm, train_tgt_px, val_feat_px_norm, val_tgt_px,
        test_feat_px_norm, test_tgt_px, geo_mean, geo_std,
        tag="MLP-pixel-only", device=device,
    )
    all_results["MLP-pixel-only"] = results_px
    print(f"\n  Mean R²: {results_px['mean_r2']:.4f}")
    for name in TARGET_NAMES:
        r = results_px[name]
        print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # Experiment 3: Label efficiency sweep
    print("\n" + "=" * 70)
    print("Experiment 3: Label efficiency sweep")
    print("=" * 70)
    for frac in [0.5, 0.25, 0.1, 0.05]:
        n_sub = int(len(train_feat) * frac)
        idx = np.random.permutation(len(train_feat))[:n_sub]
        tag = f"MLP-{int(frac*100)}pct"

        results_frac = train_and_evaluate(
            train_feat[idx], train_tgt[idx], val_feat, val_tgt,
            test_feat, test_tgt, geo_mean, geo_std,
            tag=tag, device=device,
        )
        all_results[tag] = results_frac
        print(f"\n  [{tag}] Mean R²: {results_frac['mean_r2']:.4f}")

    # Save all results
    results_path = OUTPUT_DIR / "results_mlp_geometry_baseline.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20s} {'R²':>8s} {'Params':>8s} {'Train':>10s}")
    print("-" * 50)
    for tag in ["MLP-pixel-only", "MLP-100pct", "MLP-50pct", "MLP-25pct",
                "MLP-10pct", "MLP-5pct"]:
        if tag in all_results:
            r = all_results[tag]
            print(f"{tag:<20s} {r['mean_r2']:>8.4f} {r['n_params']:>8,d} {r['n_train']:>10,d}")

    # Compare with ConvNextUNet
    print("-" * 50)
    cnn_path = OUTPUT_DIR / "results_geom_G-B.json"
    if cnn_path.exists():
        with open(cnn_path) as f:
            cnn = json.load(f)
        r2_cnn = cnn.get("test", {}).get("mean_r2", cnn.get("mean_r2", "N/A"))
        print(f"{'ConvNextUNet (G-B)':<20s} {r2_cnn:>8.4f} {'14M':>8s} {'566K':>10s}")

    gqe_path = OUTPUT_DIR / "results_uncertainty_G-QE.json"
    if gqe_path.exists():
        with open(gqe_path) as f:
            gqe = json.load(f)
        if "mean_r2" in gqe:
            print(f"{'G-QE (5×ensemble)':<20s} {gqe['mean_r2']:>8.4f} {'70M':>8s} {'566K':>10s}")


if __name__ == "__main__":
    main()
