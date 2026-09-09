"""Train dense cloud geometry prediction with quantile regression + physics ordering.

Supports three experiment modes:
  G-B:  Baseline point estimate (L1 loss, 8 outputs)
  G-Q:  Quantile regression (pinball loss, 8×3=24 outputs)
  G-QP: Quantile + physics ordering constraints

Usage:
    # Baseline point estimate
    python -u 47_train_geometry_quantile.py --tag G-B --mode point --seed 42

    # Quantile regression only
    python -u 47_train_geometry_quantile.py --tag G-Q --mode quantile --seed 42

    # Quantile + physics ordering
    python -u 47_train_geometry_quantile.py --tag G-QP --mode quantile --lambda-order 0.1 --seed 42

    # 5-seed ensemble of G-QP
    python -u 47_train_geometry_quantile.py --tag G-QP-E1 --mode quantile --lambda-order 0.1 --seed 42
    python -u 47_train_geometry_quantile.py --tag G-QP-E2 --mode quantile --lambda-order 0.1 --seed 123
    ...

    # Evaluate ensemble
    python -u 47_train_geometry_quantile.py --evaluate-ensemble G-QP-E1,G-QP-E2,...
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
from models.quantile_loss import pinball_loss, ordering_loss

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
TARGET_IDX = {name: i for i, name in enumerate(TARGET_NAMES)}
N_TARGETS = len(TARGET_NAMES)


class DenseGeometryDataset(Dataset):
    """Dense dataset with pre-extracted cloud geometry targets."""

    def __init__(self, split="train", augment=False, use_era5=True):
        self.augment = augment
        self.patches = np.load(DENSE_DIR / f"{split}_patches.npy")
        self.geometry = np.load(DENSE_DIR / f"{split}_geometry.npy")  # (N, 64, 8)
        self.positions = np.load(DENSE_DIR / f"{split}_positions.npy")
        self.n_profiles = np.load(DENSE_DIR / f"{split}_n_profiles.npy")
        self.n_samples = len(self.patches)

        # Normalization stats (computed from training set)
        stats = np.load(OUTPUT_DIR / "geometry_stats.npz")
        self.geo_mean = stats["mean"].astype(np.float32)  # (8,)
        self.geo_std = stats["std"].astype(np.float32)     # (8,)

        print(f"  {split}: {self.n_samples:,} patches, geometry shape {self.geometry.shape}")

        # ERA5
        self.use_era5 = use_era5
        if use_era5:
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
        else:
            self.era5_dim = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        patch = torch.from_numpy(np.ascontiguousarray(self.patches[idx]))  # (10, 64, 64)
        n = int(self.n_profiles[idx])

        # Build geometry target map (T, 64, 64) and mask (64, 64)
        target = torch.zeros(N_TARGETS, 64, 64)
        mask = torch.zeros(64, 64)

        if n > 0:
            rows = self.positions[idx, :n, 0].astype(np.int64)
            cols = self.positions[idx, :n, 1].astype(np.int64)
            valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
            rows, cols = rows[valid], cols[valid]

            geo = self.geometry[idx, :n][valid]  # (n_valid, 8)
            # Z-score normalize
            geo_norm = (geo - self.geo_mean) / self.geo_std
            target[:, rows, cols] = torch.from_numpy(geo_norm.T)
            mask[rows, cols] = 1.0

        # D4 augmentation
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

        if self.use_era5:
            era5 = torch.from_numpy(self.era5_flat[idx])
            era5_valid = torch.tensor(self.era5_valid[idx])
        else:
            era5 = torch.zeros(1)
            era5_valid = torch.tensor(0.0)

        return patch, target, mask, era5, era5_valid


def masked_l1_loss(pred, target, mask):
    """Masked L1 loss for point estimates. pred/target: (B, T, H, W)."""
    mask_3d = mask.unsqueeze(1)  # (B, 1, H, W)
    diff = torch.abs(pred - target) * mask_3d
    n_sup = mask_3d.sum() * pred.shape[1] + 1e-8
    return diff.sum() / n_sup


def tta_predict(model, patch, era5, era5_valid, quantile_mode=False):
    """Test-time augmentation: 4 rotations × 2 flips = 8 predictions."""
    preds = []
    for k in range(4):
        for flip in [False, True]:
            x = torch.rot90(patch, k, [2, 3])
            if flip:
                x = x.flip(-1)

            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, era5, era5_valid)
            pred = pred.float()

            # Reverse augmentation
            if flip:
                pred = pred.flip(-1)
            if k > 0:
                if quantile_mode:
                    pred = torch.rot90(pred, -k, [3, 4])  # (B, T, Q, H, W)
                else:
                    pred = torch.rot90(pred, -k, [2, 3])  # (B, T, H, W)
            preds.append(pred)

    return torch.stack(preds).mean(0)


def evaluate(model, loader, device, geo_mean, geo_std, use_era5=True,
             use_tta=False, quantile_mode=False):
    """Evaluate per-target R² and MAE."""
    model.eval()
    all_pred = []  # (N_sup, 8)
    all_true = []  # (N_sup, 8)

    with torch.no_grad():
        for batch in loader:
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]
            if not use_era5:
                era5, era5_valid = None, None

            if use_tta:
                pred = tta_predict(model, patch, era5, era5_valid, quantile_mode)
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(patch, era5, era5_valid)
                pred = pred.float()

            # If quantile mode, take median (index 1)
            if quantile_mode:
                pred = pred[:, :, 1, :, :]  # (B, T, H, W)

            pred = pred.cpu()
            target = target.float().cpu()
            mask = mask.cpu()

            B = pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                # Denormalize
                pred_geo = pred[b, :, rows, cols].T.numpy() * geo_std + geo_mean  # (n, 8)
                true_geo = target[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                all_pred.append(pred_geo)
                all_true.append(true_geo)

    all_pred = np.concatenate(all_pred, axis=0)  # (N_total, 8)
    all_true = np.concatenate(all_true, axis=0)

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = float(r2_score(all_true[:, i], all_pred[:, i]))
        mae = float(np.abs(all_true[:, i] - all_pred[:, i]).mean())
        results[name] = {"r2": r2, "mae": mae}

    # Mean R² across all targets
    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["n_pixels"] = len(all_pred)
    return results


def evaluate_ensemble(tags, use_tta=True):
    """Load multiple models, average predictions, compute R²."""
    device = torch.device("cuda")

    first_ckpt = torch.load(MODEL_DIR / f"geom_{tags[0]}.pt",
                            map_location="cpu", weights_only=False)
    quantile_mode = first_ckpt.get("quantile_mode", False)
    physics_head = first_ckpt.get("physics_head", False)
    base_dim = first_ckpt["base_dim"]
    era5_dim = first_ckpt["era5_dim"]
    geo_mean = np.array(first_ckpt["geo_mean"])
    geo_std = np.array(first_ckpt["geo_std"])

    test_ds = DenseGeometryDataset("test", augment=False)
    # Use train normalization stats from checkpoint
    test_ds.geo_mean = geo_mean
    test_ds.geo_std = geo_std
    test_ds.era5_mean = np.array(first_ckpt["era5_mean"])
    test_ds.era5_std_norm = np.array(first_ckpt["era5_std"])
    # Re-normalize ERA5
    era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    test_ds.era5_flat = era5_raw.reshape(len(era5_raw), -1)
    test_ds.era5_flat = (test_ds.era5_flat - test_ds.era5_mean) / test_ds.era5_std_norm
    valid_mask = test_ds.era5_valid > 0.5
    test_ds.era5_flat[~valid_mask] = 0.0

    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, **kwargs)

    # Load models
    models = []
    for tag in tags:
        ckpt = torch.load(MODEL_DIR / f"geom_{tag}.pt",
                          map_location="cpu", weights_only=False)
        model = ConvNextUNet(
            in_channels=10, out_channels=N_TARGETS,
            base_dim=base_dim, dim_mults=(1, 2, 4),
            era5_dim=era5_dim,
            quantile_mode=quantile_mode,
            physics_head=physics_head,
        ).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        models.append(model)
        print(f"  Loaded {tag}")

    # Ensemble prediction
    all_pred = []
    all_true = []

    print(f"\n  Evaluating ensemble of {len(models)} models (TTA={use_tta})...")
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            patch, target, mask, era5, era5_valid = [b.to(device) for b in batch]

            ensemble_pred = None
            for model in models:
                if use_tta:
                    pred = tta_predict(model, patch, era5, era5_valid, quantile_mode)
                else:
                    with autocast("cuda", dtype=torch.bfloat16):
                        pred = model(patch, era5, era5_valid)
                    pred = pred.float()

                # If quantile, take median
                if quantile_mode:
                    pred = pred[:, :, 1, :, :]

                if ensemble_pred is None:
                    ensemble_pred = pred.cpu()
                else:
                    ensemble_pred += pred.cpu()

            ensemble_pred /= len(models)
            target = target.float().cpu()
            mask = mask.cpu()

            B = ensemble_pred.shape[0]
            for b in range(B):
                rows, cols = torch.where(mask[b] > 0.5)
                if len(rows) == 0:
                    continue
                pred_geo = ensemble_pred[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                true_geo = target[b, :, rows, cols].T.numpy() * geo_std + geo_mean
                all_pred.append(pred_geo)
                all_true.append(true_geo)

            if (bi + 1) % 100 == 0:
                n_total = sum(len(p) for p in all_pred)
                print(f"    [{bi+1}/{len(test_loader)}] {n_total:,} pixels")

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = float(r2_score(all_true[:, i], all_pred[:, i]))
        mae = float(np.abs(all_true[:, i] - all_pred[:, i]).mean())
        results[name] = {"r2": r2, "mae": mae}

    results["mean_r2"] = float(np.mean([results[n]["r2"] for n in TARGET_NAMES]))
    results["n_pixels"] = len(all_pred)
    results["n_models"] = len(models)
    results["use_tta"] = use_tta
    results["tags"] = tags
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="G-B")
    parser.add_argument("--mode", choices=["point", "quantile"], default="point")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--drop-path", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lambda-order", type=float, default=0.0,
                        help="Physics ordering constraint weight (0=disabled, 0.1=recommended)")
    parser.add_argument("--lambda-order-ramp", type=int, default=10,
                        help="Epochs to ramp lambda_order from 0 to target")
    parser.add_argument("--physics-head", action="store_true",
                        help="Use PhysicsConstrainedHead (hard ordering, no loss needed)")
    parser.add_argument("--evaluate-ensemble", type=str, default=None,
                        help="Comma-separated tags to evaluate as ensemble")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-era5", action="store_true",
                        help="Train without ERA5 conditioning (ablation)")
    args = parser.parse_args()

    # Ensemble evaluation mode
    if args.evaluate_ensemble:
        tags = args.evaluate_ensemble.split(",")
        print(f"Evaluating ensemble: {tags}")
        results = evaluate_ensemble(tags, use_tta=not args.no_tta)
        print(f"\n  Mean R²: {results['mean_r2']:.4f}")
        for name in TARGET_NAMES:
            r = results[name]
            print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")
        results_path = OUTPUT_DIR / f"results_geom_ensemble_{'_'.join(tags)}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {results_path}")
        return

    quantile_mode = args.mode == "quantile"

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    print(f"Training {args.tag} (mode={args.mode}, seed={args.seed})")
    print(f"Config: base_dim={args.base_dim}, dropout={args.dropout}, "
          f"drop_path={args.drop_path}, lambda_order={args.lambda_order}")

    # Data
    use_era5 = not args.no_era5
    train_ds = DenseGeometryDataset("train", augment=True, use_era5=use_era5)
    val_ds = DenseGeometryDataset("val", augment=False, use_era5=use_era5)

    # Use train normalization for val
    val_ds.geo_mean = train_ds.geo_mean
    val_ds.geo_std = train_ds.geo_std
    if use_era5:
        val_ds.era5_mean = train_ds.era5_mean
        val_ds.era5_std_norm = train_ds.era5_std_norm
        # Re-normalize val ERA5 with train stats
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

    # Model
    model = ConvNextUNet(
        in_channels=10, out_channels=N_TARGETS,
        base_dim=args.base_dim, dim_mults=(1, 2, 4),
        era5_dim=train_ds.era5_dim, drop_path_rate=args.drop_path,
        dropout=args.dropout,
        quantile_mode=quantile_mode,
        physics_head=args.physics_head,
    ).to(device)
    print(f"Model: {count_params(model)/1e6:.2f}M params, quantile_mode={quantile_mode}, "
          f"physics_head={args.physics_head}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print(f"\nTraining: {args.epochs} ep, batch={args.batch_size}, lr={args.lr}")
    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    print("=" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        loss_main_sum, loss_ord_sum, n_batches = 0, 0, 0

        # Ramp ordering constraint weight
        if args.lambda_order > 0 and args.lambda_order_ramp > 0:
            ramp = min(1.0, epoch / args.lambda_order_ramp)
            lambda_order = args.lambda_order * ramp
        else:
            lambda_order = args.lambda_order

        for patch, target, mask, era5, era5_valid in train_loader:
            patch = patch.to(device)
            target = target.to(device)
            mask = mask.to(device)
            era5 = era5.to(device)
            era5_valid = era5_valid.to(device)

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(patch, era5, era5_valid)

                if quantile_mode:
                    loss_main = pinball_loss(pred, target, mask)
                else:
                    loss_main = masked_l1_loss(pred, target, mask)

                loss = loss_main

                if lambda_order > 0 and quantile_mode and not args.physics_head:
                    loss_ord = ordering_loss(pred, TARGET_IDX)
                    loss = loss + lambda_order * loss_ord
                    loss_ord_sum += loss_ord.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_main_sum += loss_main.item()
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
                    if quantile_mode:
                        val_loss = pinball_loss(pred, target, mask)
                    else:
                        val_loss = masked_l1_loss(pred, target, mask)
                val_loss_sum += val_loss.item()
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
            avg_main = loss_main_sum / n_batches
            parts = [f"loss={avg_main:.5f}"]
            if lambda_order > 0:
                avg_ord = loss_ord_sum / n_batches
                parts.append(f"ord={avg_ord:.4f}")
                parts.append(f"λ={lambda_order:.3f}")
            lr = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch:3d}: {' '.join(parts)} val={val_loss:.5f} "
                  f"lr={lr:.1e} [{elapsed:.0f}s]{marker}")

        if wait >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate best model
    print(f"\n{'='*70}")
    print("Evaluating best model...")
    model.load_state_dict(best_state)
    model = model.to(device).eval()

    test_ds = DenseGeometryDataset("test", augment=False, use_era5=use_era5)
    test_ds.geo_mean = train_ds.geo_mean
    test_ds.geo_std = train_ds.geo_std
    if use_era5:
        test_ds.era5_mean = train_ds.era5_mean
        test_ds.era5_std_norm = train_ds.era5_std_norm
        test_era5 = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
        nan_mask = np.isnan(test_era5)
        test_era5[nan_mask] = 0.0
        test_ds.era5_flat = test_era5.reshape(len(test_era5), -1)
        test_ds.era5_flat = (test_ds.era5_flat - train_ds.era5_mean) / train_ds.era5_std_norm
        valid_mask = test_ds.era5_valid > 0.5
        test_ds.era5_flat[~valid_mask] = 0.0

    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **kwargs)

    geo_mean = train_ds.geo_mean
    geo_std = train_ds.geo_std

    # No TTA
    results = evaluate(model, test_loader, device, geo_mean, geo_std,
                       use_tta=False, quantile_mode=quantile_mode)
    print(f"\n  Results (no TTA): mean R²={results['mean_r2']:.4f}")
    for name in TARGET_NAMES:
        r = results[name]
        print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # With TTA
    results_tta = evaluate(model, test_loader, device, geo_mean, geo_std,
                           use_tta=True, quantile_mode=quantile_mode)
    print(f"\n  Results (TTA-8): mean R²={results_tta['mean_r2']:.4f}")
    for name in TARGET_NAMES:
        r = results_tta[name]
        print(f"    {name:>12s}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # Save checkpoint
    save_path = MODEL_DIR / f"geom_{args.tag}.pt"
    ckpt = {
        "model_state_dict": best_state,
        "args": vars(args),
        "quantile_mode": quantile_mode,
        "physics_head": args.physics_head,
        "base_dim": args.base_dim,
        "era5_dim": train_ds.era5_dim,
        "geo_mean": geo_mean.tolist(),
        "geo_std": geo_std.tolist(),
        "target_names": TARGET_NAMES,
    }
    if use_era5:
        ckpt["era5_mean"] = train_ds.era5_mean.tolist()
        ckpt["era5_std"] = train_ds.era5_std_norm.tolist()
    torch.save(ckpt, save_path)

    results_all = {
        "tag": args.tag,
        "mode": args.mode,
        "seed": args.seed,
        "n_params": count_params(model),
        "best_val_loss": best_val_loss,
        "lambda_order": args.lambda_order,
        "test": results,
        "test_tta": results_tta,
    }
    results_path = OUTPUT_DIR / f"results_geom_{args.tag}.json"
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\n  Model saved: {save_path}")
    print(f"  Results: {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
