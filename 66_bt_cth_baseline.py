"""BT-based cloud-top height baseline (operational algorithm proxy).

Simulates what operational cloud products do:
1. Use 11µm brightness temperature (M15) as cloud-top temperature
2. Use ERA5 temperature profile to find matching pressure level
3. Compare with EarthCARE-derived cloud_top level target

This provides a fair physics-based baseline showing what's achievable
without ML, using the same input data (VIIRS + ERA5).

Usage:
    python -u 66_bt_cth_baseline.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.metrics import r2_score

from config import COLOC_DIR, OUTPUT_DIR

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]

# ERA5 pressure levels (26 levels, hPa) — standard set
ERA5_LEVELS = np.array([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700
], dtype=np.float32)

# EarthCARE ATL_ICE_2A has 242 levels spanning ~0-16km
# Active levels are 69-228 (159 levels)
# Level 69 ≈ surface, level 228 ≈ ~16km top
# Approximate mapping: level_index maps roughly linearly to altitude
# Level 69 ≈ 0 km, level 228 ≈ 16 km
# In our targets (0-158 active level index): 0 = lowest, 158 = highest
ACTIVE_START = 69
ACTIVE_END = 228
N_ACTIVE = ACTIVE_END - ACTIVE_START  # 159


def bt_to_cloud_top_level(patches, era5_raw, era5_valid, positions, n_profiles, geometry,
                          bt_mean=258.0, bt_std=20.0):
    """Estimate cloud-top level from BT + ERA5 temperature profile.

    BT is z-scored: bt_raw = bt_norm * bt_std + bt_mean
    ERA5 temperature is raw (in K), stored as (N, 26, 4) with T at index 0.
    """
    M15_IDX = 3
    T_IDX = 0

    all_pred_top = []
    all_true_top = []

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

        # Denormalize M15 BT, skip fill values
        bt_norm = patches[i][M15_IDX, rows, cols]
        good = bt_norm > -10
        if good.sum() == 0:
            continue
        bt_m15 = bt_norm[good] * bt_std + bt_mean  # Kelvin

        true_top = geometry[i, :n, 1][valid][good]

        if not era5_valid[i]:
            continue  # skip patches without ERA5

        # ERA5 temperature profile (26 levels, ordered top-to-bottom: 1hPa...700hPa)
        era5_T = era5_raw[i, :, T_IDX]  # (26,) in K

        for j in range(len(bt_m15)):
            bt = bt_m15[j]

            if bt >= era5_T.max():
                era5_level_frac = 25
            elif bt <= era5_T.min():
                era5_level_frac = 0
            else:
                T_reversed = era5_T[::-1]
                era5_level_frac = 12  # default
                for k in range(len(T_reversed) - 1):
                    if T_reversed[k] >= bt >= T_reversed[k + 1]:
                        frac = (T_reversed[k] - bt) / (T_reversed[k] - T_reversed[k + 1] + 1e-8)
                        era5_level_frac = 25 - (k + frac)
                        break

            era5_pressure = np.interp(era5_level_frac, np.arange(26), ERA5_LEVELS)

            if era5_pressure <= 100:
                pred_level = 158.0
            elif era5_pressure >= 700:
                pred_level = 0.0
            else:
                pred_level = 158.0 * (np.log(700) - np.log(era5_pressure)) / (np.log(700) - np.log(100))

            all_pred_top.append(pred_level)
            all_true_top.append(true_top[j])

    return np.array(all_pred_top, dtype=np.float32), np.array(all_true_top, dtype=np.float32)


def simple_bt_baseline(patches, positions, n_profiles, geometry,
                       bt_mean=258.0, bt_std=20.0):
    """BT-only cloud-top estimate (no ERA5).

    Uses all 5 thermal channels (M12-M16) with data-fitted quadratic mapping.
    BT is z-scored: bt_raw = bt_norm * bt_std + bt_mean
    """
    all_bt, all_true = [], []

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

        # M15 BT (channel 3), denormalize — skip fill values (<-10)
        bt_norm = patches[i][3, rows, cols]
        good = bt_norm > -10
        if good.sum() == 0:
            continue
        bt = bt_norm[good] * bt_std + bt_mean  # back to Kelvin

        true_top = geometry[i, :n, 1][valid][good]
        all_bt.append(bt)
        all_true.append(true_top)

    all_bt = np.concatenate(all_bt)
    all_true = np.concatenate(all_true)

    # Fit quadratic BT→cloud_top from data (best simple physics baseline)
    coeffs = np.polyfit(all_bt, all_true, 2)
    pred = np.clip(np.polyval(coeffs, all_bt), 0, 158)
    return pred, all_true


def main():
    print("Loading test data...")
    patches = np.load(DENSE_DIR / "test_patches.npy")
    geometry = np.load(DENSE_DIR / "test_geometry.npy")
    positions = np.load(DENSE_DIR / "test_positions.npy")
    n_profiles = np.load(DENSE_DIR / "test_n_profiles.npy")

    era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    era5_valid = ~nan_mask[:, 0, 0]

    print(f"  Patches: {patches.shape}, Geometry: {geometry.shape}")
    print(f"  ERA5 valid: {era5_valid.sum()}/{len(era5_valid)}")

    # Check BT range to validate channel assignment
    all_bt = []
    for i in range(min(100, len(patches))):
        n = int(n_profiles[i])
        if n > 0:
            rows = positions[i, :n, 0].astype(np.int64)
            cols = positions[i, :n, 1].astype(np.int64)
            valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
            bt = patches[i][3, rows[valid], cols[valid]]
            all_bt.append(bt)
    all_bt = np.concatenate(all_bt)
    print(f"  M15 BT range: [{all_bt.min():.1f}, {all_bt.max():.1f}] (expected ~200-310 K)")

    # Check if BT values are normalized or raw
    if all_bt.max() < 5.0:
        print("  WARNING: BT appears normalized, not in Kelvin. Checking all channels...")
        for ch in range(10):
            vals = patches[:100, ch, :, :].ravel()
            print(f"    Channel {ch}: [{vals.min():.3f}, {vals.max():.3f}]")

    # Baseline 1: Simple BT mapping
    print("\n" + "=" * 70)
    print("Baseline 1: Simple BT → cloud_top mapping")
    print("=" * 70)

    pred_bt, true_bt = simple_bt_baseline(patches, positions, n_profiles, geometry)
    r2_bt = r2_score(true_bt, pred_bt)
    mae_bt = np.abs(true_bt - pred_bt).mean()
    print(f"  cloud_top R² = {r2_bt:.4f}, MAE = {mae_bt:.2f} levels")
    print(f"  N = {len(pred_bt):,} pixels")

    results = {
        "BT_simple": {
            "cloud_top_r2": float(r2_bt),
            "cloud_top_mae": float(mae_bt),
            "n_pixels": len(pred_bt),
            "description": "Quadratic BT→level mapping (data-fitted, single channel)",
        },
    }

    # Baseline 2: Multi-channel BT regression (all 5 thermal + SZA)
    print("\n" + "=" * 70)
    print("Baseline 2: Multi-channel BT regression (5 thermal + SZA)")
    print("=" * 70)

    # Also load train set to fit the regression
    train_patches = np.load(DENSE_DIR / "train_patches.npy")
    train_geometry = np.load(DENSE_DIR / "train_geometry.npy")
    train_positions = np.load(DENSE_DIR / "train_positions.npy")
    train_n_profiles = np.load(DENSE_DIR / "train_n_profiles.npy")

    def extract_bt_features(patches, positions, n_profiles, geometry):
        all_feat, all_tgt = [], []
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
            # 5 thermal channels (0-4) + SZA (9), skip fill values
            feats = patches[i][[0,1,2,3,4,9], :, :][:, rows, cols].T  # (n, 6)
            good = feats[:, 3] > -10  # M15 not fill
            if good.sum() == 0:
                continue
            all_feat.append(feats[good])
            all_tgt.append(geometry[i, :n, 1][valid][good])
        return np.concatenate(all_feat), np.concatenate(all_tgt)

    train_feat, train_tgt = extract_bt_features(
        train_patches, train_positions, train_n_profiles, train_geometry)
    test_feat, test_tgt = extract_bt_features(
        patches, positions, n_profiles, geometry)
    print(f"  Train: {len(train_feat):,}, Test: {len(test_feat):,}")

    # Ridge regression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_poly = poly.fit_transform(train_feat)
    test_poly = poly.transform(test_feat)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_poly, train_tgt)
    pred_ridge = np.clip(ridge.predict(test_poly), 0, 158)
    r2_ridge = r2_score(test_tgt, pred_ridge)
    mae_ridge = np.abs(test_tgt - pred_ridge).mean()
    print(f"  cloud_top R² = {r2_ridge:.4f}, MAE = {mae_ridge:.2f} levels")
    print(f"  N = {len(pred_ridge):,} pixels")

    results["BT_multiband"] = {
        "cloud_top_r2": float(r2_ridge),
        "cloud_top_mae": float(mae_ridge),
        "n_pixels": len(pred_ridge),
        "description": "Quadratic ridge regression on 5 thermal BTs + SZA",
    }

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON (cloud_top only)")
    print("=" * 70)

    # results dict is initialized above, after Baseline 1

    # Load ML results for comparison
    for tag in ["G-B", "G-QE"]:
        p = OUTPUT_DIR / f"results_geom_{tag}.json"
        if not p.exists():
            p = OUTPUT_DIR / f"results_uncertainty_{tag}.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            ct_r2 = None
            if "test" in r and "cloud_top" in r["test"]:
                ct_r2 = r["test"]["cloud_top"]["r2"]
            elif "per_target" in r and "cloud_top" in r["per_target"]:
                ct_r2 = r["per_target"]["cloud_top"]["r2"]
            elif "cloud_top" in r:
                ct_r2 = r["cloud_top"]["r2"]
            if ct_r2:
                results[tag] = {"cloud_top_r2": ct_r2}

    # MLP baseline
    mlp_p = OUTPUT_DIR / "results_mlp_geometry_baseline.json"
    if mlp_p.exists():
        with open(mlp_p) as f:
            mlp = json.load(f)
        if "MLP-100pct" in mlp:
            results["MLP"] = {
                "cloud_top_r2": mlp["MLP-100pct"]["cloud_top"]["r2"]
            }

    print(f"{'Method':<25s} {'cloud_top R²':>12s}")
    print("-" * 40)
    for tag in ["BT_simple", "BT_multiband", "MLP", "G-B", "G-QE"]:
        if tag in results and "cloud_top_r2" in results[tag]:
            print(f"{tag:<25s} {results[tag]['cloud_top_r2']:>12.4f}")

    results_path = OUTPUT_DIR / "results_bt_baseline.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
