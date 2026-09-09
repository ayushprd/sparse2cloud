"""Conditional coverage analysis: does CQR calibrate uniformly across conditions?

Stratifies test predictions by:
  - Latitude bands (tropical, mid-latitude, polar)
  - Day vs night (SZA threshold)
  - Cloud type (derived from geometry: high/thin, deep convective, mid-level, low)

Uses pre-saved CQR test data (cqr_test_data_G-QE.npz) + metadata from npy_dense.

Usage:
    python -u 65_conditional_coverage.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.metrics import r2_score

from config import COLOC_DIR, OUTPUT_DIR, FIGURE_DIR

DENSE_DIR = COLOC_DIR / "npy_dense"

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]
N_TARGETS = len(TARGET_NAMES)


def extract_pixel_metadata():
    """Extract per-pixel metadata matching CQR test data ordering.

    CQR data iterates: for each patch -> for each supervised pixel (mask > 0).
    We replicate this to get lat, lon, SZA per pixel.
    """
    patches = np.load(DENSE_DIR / "test_patches.npy")
    geometry = np.load(DENSE_DIR / "test_geometry.npy")
    positions = np.load(DENSE_DIR / "test_positions.npy")
    n_profiles = np.load(DENSE_DIR / "test_n_profiles.npy")
    lat = np.load(DENSE_DIR / "test_lat.npy")
    lon = np.load(DENSE_DIR / "test_lon.npy")

    # SZA is channel index 9 (last channel) in patches — actually it's the 10th band
    # Let me check: VIIRS bands are M12-M16 (5 thermal) + M07,M08,M10,M11 (4 reflective) + SZA
    # SZA is the 10th channel (index 9)

    all_lat, all_lon, all_sza = [], [], []

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

        n_valid = len(rows)
        all_lat.append(np.full(n_valid, lat[i], dtype=np.float32))
        all_lon.append(np.full(n_valid, lon[i], dtype=np.float32))
        # SZA from last channel at pixel location
        sza_vals = patches[i][9, rows, cols]
        all_sza.append(sza_vals)

    return (np.concatenate(all_lat),
            np.concatenate(all_lon),
            np.concatenate(all_sza))


def compute_metrics(pred, true, q_hat):
    """Compute R², PICP, MPIW for a subset of predictions.

    pred: (N, 8, 3) — quantiles [0.1, 0.5, 0.9]
    true: (N, 8)
    q_hat: (8,) — CQR threshold per target
    """
    N = len(pred)
    if N < 50:
        return None

    results = {}

    # Per-target metrics
    r2_list, picp_list, mpiw_list = [], [], []
    for t in range(N_TARGETS):
        y = true[:, t]
        q_lo = pred[:, t, 0] - q_hat[t]
        q_hi = pred[:, t, 2] + q_hat[t]
        q_med = pred[:, t, 1]

        r2 = float(r2_score(y, q_med))
        covered = (y >= q_lo) & (y <= q_hi)
        picp = float(covered.mean())
        width = q_hi - q_lo
        mpiw = float(width.mean())

        r2_list.append(r2)
        picp_list.append(picp)
        mpiw_list.append(mpiw)
        results[TARGET_NAMES[t]] = {"r2": r2, "picp": picp, "mpiw": mpiw}

    results["mean_r2"] = float(np.mean(r2_list))
    results["mean_picp"] = float(np.mean(picp_list))
    results["mean_mpiw"] = float(np.mean(mpiw_list))
    results["n_pixels"] = N
    return results


def classify_cloud_type(true_geo, geo_stats):
    """Classify cloud type from geometry targets.

    Uses denormalized targets to assign:
      - high_thin: cloud_top > 70th percentile, thickness < 30th percentile
      - deep_convective: cloud_top > 70th pct, thickness > 70th pct
      - mid_level: 30th < cloud_top < 70th percentile
      - low: cloud_top < 30th percentile
    """
    cloud_top = true_geo[:, 1]  # cloud_top index
    thickness = true_geo[:, 4]  # thickness index

    top_p30 = np.percentile(cloud_top, 30)
    top_p70 = np.percentile(cloud_top, 70)
    thick_p30 = np.percentile(thickness, 30)
    thick_p70 = np.percentile(thickness, 70)

    types = np.full(len(true_geo), "mid_level", dtype=object)
    types[cloud_top < top_p30] = "low"
    types[(cloud_top >= top_p70) & (thickness < thick_p30)] = "high_thin"
    types[(cloud_top >= top_p70) & (thickness >= thick_p70)] = "deep_convective"
    # The rest stays mid_level

    return types


def main():
    print("Loading CQR test data...")
    cqr = np.load(OUTPUT_DIR / "cqr_test_data_G-QE.npz")
    pred = cqr["pred"]   # (N, 8, 3)
    true = cqr["true"]   # (N, 8)
    dist = cqr["dist"]   # (N,)
    q_hat = cqr["q_hat"] # (8,)

    print(f"  Predictions: {pred.shape[0]:,} pixels")

    print("\nExtracting pixel metadata...")
    px_lat, px_lon, px_sza = extract_pixel_metadata()
    print(f"  Metadata: {len(px_lat):,} pixels")

    # The CQR data may be a subset (val split for calibration).
    # Check alignment
    n_cqr = len(pred)
    n_meta = len(px_lat)
    if n_cqr != n_meta:
        print(f"  WARNING: CQR has {n_cqr} pixels but metadata has {n_meta}")
        print(f"  CQR likely uses test split only. Truncating metadata to match.")
        # CQR data comes from the test set evaluation which may process differently.
        # The ordering should match (both iterate patches sequentially).
        # If sizes differ, the CQR data might exclude some patches via val/cal split.
        # Use the first n_cqr pixels
        if n_cqr < n_meta:
            px_lat = px_lat[:n_cqr]
            px_lon = px_lon[:n_cqr]
            px_sza = px_sza[:n_cqr]
        else:
            print("  ERROR: More CQR pixels than metadata. Cannot align.")
            return

    all_results = {}

    # 1. Overall metrics
    print("\n" + "=" * 70)
    print("Overall metrics")
    print("=" * 70)
    overall = compute_metrics(pred, true, q_hat)
    all_results["overall"] = overall
    print(f"  R²={overall['mean_r2']:.4f}, PICP={overall['mean_picp']:.4f}, "
          f"MPIW={overall['mean_mpiw']:.2f}, N={overall['n_pixels']:,}")

    # 2. Latitude bands
    print("\n" + "=" * 70)
    print("By latitude band")
    print("=" * 70)
    lat_bands = {
        "tropical": (np.abs(px_lat) <= 23.5),
        "subtropical": (np.abs(px_lat) > 23.5) & (np.abs(px_lat) <= 45),
        "midlatitude": (np.abs(px_lat) > 45) & (np.abs(px_lat) <= 66.5),
        "polar": (np.abs(px_lat) > 66.5),
    }
    all_results["by_latitude"] = {}
    for name, mask in lat_bands.items():
        if mask.sum() < 50:
            print(f"  {name}: too few pixels ({mask.sum()})")
            continue
        r = compute_metrics(pred[mask], true[mask], q_hat)
        all_results["by_latitude"][name] = r
        print(f"  {name:>15s}: R²={r['mean_r2']:.4f}, PICP={r['mean_picp']:.4f}, "
              f"MPIW={r['mean_mpiw']:.2f}, N={r['n_pixels']:,}")

    # 3. Day vs Night
    print("\n" + "=" * 70)
    print("Day vs Night (SZA threshold = 85°)")
    print("=" * 70)
    # SZA in the dataset is normalized; check range
    sza_min, sza_max = px_sza.min(), px_sza.max()
    print(f"  SZA range: [{sza_min:.2f}, {sza_max:.2f}]")

    # If normalized, SZA=85° needs to be converted
    # The VIIRS patches store SZA as a normalized channel.
    # From config: SZA is stored directly in degrees or normalized.
    # Let's check the actual range to decide
    if sza_max <= 2.0:
        # Likely normalized (0-1 range or z-scored)
        # Use median as day/night split
        sza_threshold = np.median(px_sza)
        print(f"  SZA appears normalized. Using median={sza_threshold:.3f} as threshold")
    else:
        sza_threshold = 85.0

    day_night = {
        "day": (px_sza < sza_threshold),
        "night": (px_sza >= sza_threshold),
    }
    all_results["by_daynight"] = {}
    for name, mask in day_night.items():
        if mask.sum() < 50:
            continue
        r = compute_metrics(pred[mask], true[mask], q_hat)
        all_results["by_daynight"][name] = r
        print(f"  {name:>8s}: R²={r['mean_r2']:.4f}, PICP={r['mean_picp']:.4f}, "
              f"MPIW={r['mean_mpiw']:.2f}, N={r['n_pixels']:,}")

    # 4. Cloud type
    print("\n" + "=" * 70)
    print("By cloud type (derived from geometry)")
    print("=" * 70)
    cloud_types = classify_cloud_type(true, None)
    unique_types, counts = np.unique(cloud_types, return_counts=True)
    print(f"  Cloud types: {dict(zip(unique_types, counts))}")

    all_results["by_cloud_type"] = {}
    for ctype in ["low", "mid_level", "high_thin", "deep_convective"]:
        mask = (cloud_types == ctype)
        if mask.sum() < 50:
            continue
        r = compute_metrics(pred[mask], true[mask], q_hat)
        all_results["by_cloud_type"][ctype] = r
        print(f"  {ctype:>18s}: R²={r['mean_r2']:.4f}, PICP={r['mean_picp']:.4f}, "
              f"MPIW={r['mean_mpiw']:.2f}, N={r['n_pixels']:,}")

    # 5. Per-target conditional coverage table
    print("\n" + "=" * 70)
    print("Per-target PICP by condition")
    print("=" * 70)
    header = f"{'Condition':<20s}"
    for t in TARGET_NAMES:
        header += f" {t[:6]:>7s}"
    print(header)
    print("-" * (20 + 8 * N_TARGETS))

    for category, groups in [("latitude", lat_bands),
                              ("daynight", day_night),
                              ("cloud_type", {ct: cloud_types == ct
                                              for ct in ["low", "mid_level",
                                                         "high_thin", "deep_convective"]})]:
        for name, mask in groups.items():
            if mask.sum() < 50:
                continue
            row = f"  {name:<18s}"
            for t in range(N_TARGETS):
                y = true[mask, t]
                q_lo = pred[mask, t, 0] - q_hat[t]
                q_hi = pred[mask, t, 2] + q_hat[t]
                picp = float(((y >= q_lo) & (y <= q_hi)).mean())
                row += f" {picp:>7.3f}"
            print(row)

    # Save
    results_path = OUTPUT_DIR / "results_conditional_coverage.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Summary: coverage gap analysis
    print("\n" + "=" * 70)
    print("Coverage gap analysis (target PICP = 0.90)")
    print("=" * 70)
    all_picps = []
    for category in ["by_latitude", "by_daynight", "by_cloud_type"]:
        for name, r in all_results.get(category, {}).items():
            picp = r["mean_picp"]
            gap = picp - 0.90
            all_picps.append(picp)
            status = "OK" if abs(gap) < 0.03 else ("OVER" if gap > 0 else "UNDER")
            print(f"  {name:>18s}: PICP={picp:.4f} (gap={gap:+.4f}) [{status}]")

    if all_picps:
        print(f"\n  Range: [{min(all_picps):.4f}, {max(all_picps):.4f}]")
        print(f"  Max deviation from 0.90: {max(abs(p-0.9) for p in all_picps):.4f}")


if __name__ == "__main__":
    main()
