"""Train uncertainty baseline models: heteroscedastic + evidential.

Also provides MC-Dropout and Deep Ensemble evaluation (no training needed).

Usage:
    # Heteroscedastic (beta-NLL)
    python -u 52_train_uncertainty_baselines.py --tag G-HET --mode heteroscedastic --seed 42

    # Evidential (NIG)
    python -u 52_train_uncertainty_baselines.py --tag G-EVI --mode evidential --seed 42

    # MC-Dropout evaluation (reuses G-B checkpoint)
    python -u 52_train_uncertainty_baselines.py --tag G-MCD --mode mc-dropout --base-model geom_G-B.pt --T 30

    # Deep Ensemble evaluation (reuses G-DE1..5 checkpoints)
    python -u 52_train_uncertainty_baselines.py --tag G-DE --mode ensemble --ensemble-tags G-B,G-DE2,G-DE3,G-DE4,G-DE5
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score
from pathlib import Path

from config import COLOC_DIR, OUTPUT_DIR, MODEL_DIR
from models.convnext_unet import ConvNextUNet, count_params
from models.hetero_loss import beta_nll_loss
from models.evidential_loss import evidential_loss, parse_nig

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


class DenseGeometryDataset(Dataset):
    """Dense dataset with pre-extracted cloud geometry targets."""

    def __init__(self, split="train", augment=False):
        self.augment = augment
        self.patches = np.load(DENSE_DIR / f"{split}_patches.npy")
        self.geometry = np.load(DENSE_DIR / f"{split}_geometry.npy")
        self.positions = np.load(DENSE_DIR / f"{split}_positions.npy")
        self.n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")
        self.n_samples = len(self.patches)

        stats = np.load(OUTPUT_DIR / "geometry_stats.npz")
        self.geo_mean = stats["mean"].astype(np.float32)
        self.geo_std = stats["std"].astype(np.float32)

        era5_raw = np.load(DENSE_DIR / f"{split}_era5.npy").astype(np.float32)
        nan_mask = np.isnan(era5_raw)
        era5_raw[nan_mask] = 0.0
        self.era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
        self.era5_flat = era5_raw.reshape(len(era5_raw), -1)
        valid = self.era5_valid > 0.5
        if valid.sum() > 100:
            self.era5_mean = self.era5_flat[valid].mean(axis=0)
            self.era5_std_norm = self.era5_flat[valid].std(axis=0) + 1e-8
        else:
            self.era5_mean = np.zeros(self.era5_flat.shape[1], dtype=np.float32)
            self.era5_std_norm = np.ones(self.era5_flat.shape[1], dtype=np.float32)
        self.era5_flat = (self.era5_flat - self.era5_mean) / self.era5_std_norm
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

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch = torch.rot90(patch, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])
                mask = torch.rot90(mask, k, [0, 1])
            if torch.rand(1) > 0.5:
                patch = patch.flip(-1)
                target = target.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(1) > 0.5:
                patch = patch.flip(-2)
                target = target.flip(-2)
                mask = mask.flip(-2)

        era5 = torch.from_numpy(self.era5_flat[idx])
        era5_valid = torch.tensor(self.era5_valid[idx])
        return patch, target, mask, era5, era5_valid


def extract_intervals(pred, mode, n_targets=8):
    """Extract (q_lo, q_med, q_hi) from model output. All (B, T, H, W)."""
    if mode == "heteroscedastic":
        mean = pred[:, :n_targets]
        log_var = pred[:, n_targets:].clamp(-10, 10)
        std = torch.exp(0.5 * log_var)
        return mean - 1.28 * std, mean, mean + 1.28 * std
    elif mode == "evidential":
        gamma, nu, alpha, beta = parse_nig(pred, n_targets)
        total_var = beta / (alpha - 1).clamp(min=0.01)
        std = torch.sqrt(total_var)
        return gamma - 1.28 * std, gamma, gamma + 1.28 * std
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate(model, loader, device, geo_mean, geo_std, mode):
    """Evaluate per-target R² and MAE."""
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(patch, era5, era5_valid)
            pred = pred.float().cpu()

            # Get point estimate (mean)
            if mode == "heteroscedastic":
                pred_mean = pred[:, :N_TARGETS]
            elif mode == "evidential":
                pred_mean = pred[:, :N_TARGETS]  # gamma
            else:
                pred_mean = pred

            target = target.float().cpu()
            mask = mask.cpu()

            B = pred_mean.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                p = pred_mean[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                t = target[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                all_pred.append(p)
                all_true.append(t)

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = float(r2_score(all_true[:, i], all_pred[:, i]))
        mae = float(np.abs(all_true[:, i] - all_pred[:, i]).mean())
        results[name] = {"r2": r2, "mae": mae}
    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["n_pixels"] = len(all_pred)
    return results


def mc_dropout_evaluate(model_path, test_loader, device, geo_mean, geo_std, T=30):
    """MC-Dropout: run T forward passes with dropout enabled."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ConvNextUNet(
        in_channels=10, out_channels=N_TARGETS,
        base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
        era5_dim=ckpt["era5_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Enable only Dropout2d at test time (not StochasticDepth)
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()

    all_pred, all_true, all_lo, all_hi = [], [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
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
                p = mean[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                t = target[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                s = std[b, :, rows, cols].T.numpy() * geo_std  # scale std
                all_pred.append(p)
                all_true.append(t)
                all_lo.append(p - 1.28 * s)
                all_hi.append(p + 1.28 * s)

            if (bi + 1) % 50 == 0:
                n_total = sum(len(p) for p in all_pred)
                print(f"    [{bi+1}/{len(test_loader)}] {n_total:,} pixels")

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    all_lo = np.concatenate(all_lo)
    all_hi = np.concatenate(all_hi)

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = float(r2_score(all_true[:, i], all_pred[:, i]))
        mae = float(np.abs(all_true[:, i] - all_pred[:, i]).mean())
        covered = (all_true[:, i] >= all_lo[:, i]) & (all_true[:, i] <= all_hi[:, i])
        picp = float(covered.mean())
        mpiw = float((all_hi[:, i] - all_lo[:, i]).mean())
        results[name] = {"r2": r2, "mae": mae, "raw_picp": picp, "raw_mpiw": mpiw}
    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["n_pixels"] = len(all_pred)

    # Save predictions for CQR calibration
    pred_3 = np.stack([all_lo, all_pred, all_hi], axis=2)  # (N, 8, 3)
    np.savez(OUTPUT_DIR / "uncertainty_pred_mc_dropout.npz",
             pred=pred_3, true=all_true)

    return results


def train_baseline(args):
    """Train heteroscedastic or evidential model."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    mode = args.mode
    if mode == "heteroscedastic":
        out_channels = N_TARGETS * 2  # mean + log_var
    elif mode == "evidential":
        out_channels = N_TARGETS * 4  # gamma + nu + alpha + beta
    else:
        raise ValueError(f"Training not supported for mode: {mode}")

    print(f"Training {args.tag} (mode={mode}, seed={args.seed})")

    train_ds = DenseGeometryDataset("train", augment=True)
    val_ds = DenseGeometryDataset("val", augment=False)
    val_ds.geo_mean = train_ds.geo_mean
    val_ds.geo_std = train_ds.geo_std
    val_ds.era5_mean = train_ds.era5_mean
    val_ds.era5_std_norm = train_ds.era5_std_norm
    val_era5 = np.load(DENSE_DIR / "val_era5.npy").astype(np.float32)
    nan_mask = np.isnan(val_era5)
    val_era5[nan_mask] = 0.0
    val_ds.era5_flat = val_era5.reshape(len(val_era5), -1)
    val_ds.era5_flat = (val_ds.era5_flat - train_ds.era5_mean) / train_ds.era5_std_norm
    valid_mask = val_ds.era5_valid > 0.5
    val_ds.era5_flat[~valid_mask] = 0.0

    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **kwargs)

    model = ConvNextUNet(
        in_channels=10, out_channels=out_channels,
        base_dim=args.base_dim, dim_mults=(1, 2, 4),
        era5_dim=train_ds.era5_dim, drop_path_rate=0.4,
        dropout=0.15,
    ).to(device)
    print(f"Model: {count_params(model)/1e6:.2f}M params, out_channels={out_channels}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=5e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print(f"\nTraining: {args.epochs} ep, batch={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0, 0

        for patch, target, mask, era5, era5_valid in train_loader:
            patch = patch.to(device)
            target = target.to(device)
            mask = mask.to(device)
            era5 = era5.to(device)
            era5_valid = era5_valid.to(device)

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(patch, era5, era5_valid)
                if mode == "heteroscedastic":
                    loss = beta_nll_loss(pred, target, mask, beta=0.5)
                elif mode == "evidential":
                    loss = evidential_loss(pred, target, mask, coeff=0.01)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum, val_batches = 0, 0
        with torch.no_grad():
            for patch, target, mask, era5, era5_valid in val_loader:
                patch = patch.to(device)
                target = target.to(device)
                mask = mask.to(device)
                era5 = era5.to(device)
                era5_valid = era5_valid.to(device)
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                    if mode == "heteroscedastic":
                        vl = beta_nll_loss(pred, target, mask, beta=0.5)
                    elif mode == "evidential":
                        vl = evidential_loss(pred, target, mask, coeff=0.01)
                val_loss_sum += vl.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1
            marker = f" (wait={wait})" if wait > 3 else ""

        if epoch % 5 == 0 or wait == 0 or wait >= args.patience:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch:3d}: loss={loss_sum/n_batches:.5f} val={val_loss:.5f} "
                  f"lr={lr:.1e} [{elapsed:.0f}s]{marker}")

        if wait >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate
    print(f"\n{'='*70}")
    print("Evaluating best model...")
    model.load_state_dict(best_state)
    model = model.to(device).eval()

    test_ds = DenseGeometryDataset("test", augment=False)
    test_ds.geo_mean = train_ds.geo_mean
    test_ds.geo_std = train_ds.geo_std
    test_ds.era5_mean = train_ds.era5_mean
    test_ds.era5_std_norm = train_ds.era5_std_norm
    test_era5 = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nm = np.isnan(test_era5); test_era5[nm] = 0.0
    test_ds.era5_flat = test_era5.reshape(len(test_era5), -1)
    test_ds.era5_flat = (test_ds.era5_flat - train_ds.era5_mean) / train_ds.era5_std_norm
    vm = test_ds.era5_valid > 0.5; test_ds.era5_flat[~vm] = 0.0

    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **kwargs)
    geo_mean = train_ds.geo_mean
    geo_std = train_ds.geo_std

    results = evaluate(model, test_loader, device, geo_mean, geo_std, mode)
    print(f"\n  Mean R²: {results['mean_r2']:.4f}")
    for name in TARGET_NAMES:
        r = results[name]
        print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # Save checkpoint
    save_path = MODEL_DIR / f"geom_{args.tag}.pt"
    torch.save({
        "model_state_dict": best_state,
        "args": vars(args),
        "mode": mode,
        "out_channels": out_channels,
        "base_dim": args.base_dim,
        "era5_dim": train_ds.era5_dim,
        "geo_mean": geo_mean.tolist(),
        "geo_std": geo_std.tolist(),
        "era5_mean": train_ds.era5_mean.tolist(),
        "era5_std": train_ds.era5_std_norm.tolist(),
        "target_names": TARGET_NAMES,
    }, save_path)

    results_path = OUTPUT_DIR / f"results_geom_{args.tag}.json"
    results["tag"] = args.tag
    results["mode"] = mode
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Model: {save_path}")
    print(f"  Results: {results_path}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--mode", choices=["heteroscedastic", "evidential", "mc-dropout", "ensemble"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    # MC-Dropout specific
    parser.add_argument("--base-model", type=str, default="geom_G-B.pt")
    parser.add_argument("--T", type=int, default=30)
    # Ensemble specific
    parser.add_argument("--ensemble-tags", type=str, default="")
    args = parser.parse_args()

    if args.mode in ("heteroscedastic", "evidential"):
        train_baseline(args)
    elif args.mode == "mc-dropout":
        print(f"MC-Dropout evaluation (T={args.T}) using {args.base_model}")
        device = torch.device("cuda")
        # Load test set with train stats from base model
        ckpt = torch.load(MODEL_DIR / args.base_model, map_location="cpu", weights_only=False)
        geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
        geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
        era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
        era5_std = np.array(ckpt["era5_std"], dtype=np.float32)

        test_ds = DenseGeometryDataset("test", augment=False)
        test_ds.geo_mean = geo_mean
        test_ds.geo_std = geo_std
        test_ds.era5_mean = era5_mean
        test_ds.era5_std_norm = era5_std
        test_era5 = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
        nm = np.isnan(test_era5); test_era5[nm] = 0.0
        test_ds.era5_flat = test_era5.reshape(len(test_era5), -1)
        test_ds.era5_flat = (test_ds.era5_flat - era5_mean) / era5_std
        vm = test_ds.era5_valid > 0.5; test_ds.era5_flat[~vm] = 0.0

        kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

        results = mc_dropout_evaluate(MODEL_DIR / args.base_model, test_loader,
                                       device, geo_mean, geo_std, T=args.T)
        print(f"\n  MC-Dropout (T={args.T}): mean R²={results['mean_r2']:.4f}")
        for name in TARGET_NAMES:
            r = results[name]
            print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}, "
                  f"PICP={r['raw_picp']:.4f}, MPIW={r['raw_mpiw']:.4f}")

        results_path = OUTPUT_DIR / f"results_geom_{args.tag}.json"
        results["tag"] = args.tag
        results["mode"] = "mc-dropout"
        results["T"] = args.T
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results: {results_path}")

    elif args.mode == "ensemble":
        print("Deep Ensemble evaluation")
        # Handled via 47_train_geometry_quantile.py --evaluate-ensemble
        tags = args.ensemble_tags.split(",")
        print(f"  Tags: {tags}")
        print("  Use: python -u 47_train_geometry_quantile.py --evaluate-ensemble " + ",".join(tags))


if __name__ == "__main__":
    main()
