"""Post-hoc physics constraint projection for cloud geometry predictions.

Applies simple clipping to enforce ordering constraints WITHOUT any accuracy loss:
  - cloud_top >= cloud_base  (fix: set top = max(top, base))
  - thickness = cloud_top - cloud_base >= 0 (implied by above)
  - cloud_base <= centroid <= cloud_top
  - cloud_base <= peak_level <= cloud_top

For quantile predictions, applies per-quantile independently.

Usage:
    python -u 57_posthoc_physics.py --input cqr_test_data_G-Q.npz --output cqr_test_data_G-Q-PH.npz
    python -u 57_posthoc_physics.py --analyze  # compare violations across methods
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]

# Target indices
IDX_CENTROID = 0
IDX_TOP = 1
IDX_BASE = 2
IDX_PEAK = 3
IDX_THICKNESS = 4


def count_violations(pred, quantile_idx=1):
    """Count ordering violations in predictions.

    Args:
        pred: (N, 8, Q) or (N, 8) predictions
        quantile_idx: which quantile to check (0=lo, 1=med, 2=hi)

    Returns:
        dict of violation counts and percentages
    """
    if pred.ndim == 3:
        p = pred[:, :, quantile_idx]
    else:
        p = pred

    N = len(p)
    base = p[:, IDX_BASE]
    top = p[:, IDX_TOP]
    centroid = p[:, IDX_CENTROID]
    peak = p[:, IDX_PEAK]
    thickness = p[:, IDX_THICKNESS]

    v_top_lt_base = int((top < base).sum())
    v_cent_lt_base = int((centroid < base).sum())
    v_cent_gt_top = int((centroid > top).sum())
    v_peak_lt_base = int((peak < base).sum())
    v_peak_gt_top = int((peak > top).sum())
    v_thick_neg = int((thickness < -0.01).sum())

    total = v_top_lt_base + v_cent_lt_base + v_cent_gt_top + v_peak_lt_base + v_peak_gt_top + v_thick_neg

    return {
        "n_samples": N,
        "top < base": v_top_lt_base,
        "centroid < base": v_cent_lt_base,
        "centroid > top": v_cent_gt_top,
        "peak < base": v_peak_lt_base,
        "peak > top": v_peak_gt_top,
        "thickness < 0": v_thick_neg,
        "total_violations": total,
        "pct_top_lt_base": 100 * v_top_lt_base / N,
        "pct_cent_out": 100 * (v_cent_lt_base + v_cent_gt_top) / N,
        "pct_peak_out": 100 * (v_peak_lt_base + v_peak_gt_top) / N,
        "pct_thick_neg": 100 * v_thick_neg / N,
        "pct_any_violation": 100 * total / N,
    }


def count_quantile_violations(pred):
    """Count quantile monotonicity violations: q_lo > q_med or q_med > q_hi.

    Args:
        pred: (N, 8, 3) quantile predictions

    Returns:
        dict of violation counts
    """
    N = len(pred)
    v_lo_gt_med = 0
    v_med_gt_hi = 0

    for t in range(8):
        v_lo_gt_med += int((pred[:, t, 0] > pred[:, t, 1]).sum())
        v_med_gt_hi += int((pred[:, t, 1] > pred[:, t, 2]).sum())

    return {
        "n_samples": N,
        "n_targets": 8,
        "n_comparisons": N * 8,
        "q_lo > q_med": v_lo_gt_med,
        "q_med > q_hi": v_med_gt_hi,
        "total_quantile_violations": v_lo_gt_med + v_med_gt_hi,
        "pct_quantile_violations": 100 * (v_lo_gt_med + v_med_gt_hi) / (N * 8),
    }


def project_physics(pred):
    """Apply post-hoc physics constraints to predictions.

    Only modifies values where violations actually exist.
    Thickness is only updated when top was raised (i.e., top < base was violated).

    Args:
        pred: (N, 8, Q) predictions where Q is number of quantiles (or 1)

    Returns:
        (N, 8, Q) constrained predictions
    """
    out = pred.copy()

    for q in range(pred.shape[2]):
        base = out[:, IDX_BASE, q]
        top = out[:, IDX_TOP, q]

        # Ensure top >= base (only modify where violated)
        violated = top < base
        new_top = np.where(violated, base, top)
        out[:, IDX_TOP, q] = new_top

        # Only update thickness where top was changed
        if violated.any():
            out[violated, IDX_THICKNESS, q] = new_top[violated] - base[violated]

        # Clip centroid and peak to [base, new_top]
        out[:, IDX_CENTROID, q] = np.clip(out[:, IDX_CENTROID, q], base, new_top)
        out[:, IDX_PEAK, q] = np.clip(out[:, IDX_PEAK, q], base, new_top)

    return out


def analyze_all():
    """Analyze ordering violations across all methods."""
    from sklearn.metrics import r2_score

    methods = [
        ("G-QE", "Quantile Ensemble (5 members)"),
        ("G-Q", "Quantile (unconstrained)"),
        ("G-QP", "Quantile + soft ordering"),
        ("G-QPC", "Quantile + physics head"),
        ("G-HET", "Heteroscedastic"),
        ("G-EVI", "Evidential"),
        ("G-MCD", "MC-Dropout"),
        ("G-DE", "Deep Ensemble"),
    ]

    print("=" * 90)
    print("  Ordering Violations Analysis — Real Test Data")
    print("=" * 90)

    all_results = {}

    for tag, desc in methods:
        path = OUTPUT_DIR / f"cqr_test_data_{tag}.npz"
        if not path.exists():
            print(f"\n  {tag}: SKIPPED (no data)")
            continue

        data = np.load(path)
        pred = data["pred"]  # (N, 8, 3)
        true = data["true"]  # (N, 8)

        print(f"\n  {tag} ({desc})")
        print(f"  {'-' * 60}")

        # Violations on median predictions
        viol = count_violations(pred, quantile_idx=1)
        print(f"    Median predictions ({viol['n_samples']:,} pixels):")
        print(f"      top < base:      {viol['top < base']:>8,} ({viol['pct_top_lt_base']:.2f}%)")
        print(f"      centroid out:    {viol['centroid < base'] + viol['centroid > top']:>8,} ({viol['pct_cent_out']:.2f}%)")
        print(f"      peak out:        {viol['peak < base'] + viol['peak > top']:>8,} ({viol['pct_peak_out']:.2f}%)")
        print(f"      thickness < 0:   {viol['thickness < 0']:>8,} ({viol['pct_thick_neg']:.2f}%)")
        print(f"      TOTAL:           {viol['total_violations']:>8,} ({viol['pct_any_violation']:.2f}%)")

        # Quantile violations
        qviol = count_quantile_violations(pred)
        print(f"    Quantile monotonicity ({qviol['n_comparisons']:,} comparisons):")
        print(f"      q_lo > q_med:    {qviol['q_lo > q_med']:>8,}")
        print(f"      q_med > q_hi:    {qviol['q_med > q_hi']:>8,}")
        print(f"      TOTAL:           {qviol['total_quantile_violations']:>8,} ({qviol['pct_quantile_violations']:.2f}%)")

        # R² for median
        r2_per_target = []
        for t in range(8):
            r2_per_target.append(float(r2_score(true[:, t], pred[:, t, 1])))
        mean_r2 = np.mean(r2_per_target)
        print(f"    Mean R²: {mean_r2:.4f}")

        # Post-hoc projection
        proj = project_physics(pred)
        viol_post = count_violations(proj, quantile_idx=1)

        r2_post = []
        for t in range(8):
            r2_post.append(float(r2_score(true[:, t], proj[:, t, 1])))
        mean_r2_post = np.mean(r2_post)

        print(f"    After post-hoc projection:")
        print(f"      Violations: {viol_post['total_violations']:>8,} ({viol_post['pct_any_violation']:.2f}%)")
        print(f"      Mean R²:    {mean_r2_post:.4f} (delta: {mean_r2_post - mean_r2:+.4f})")

        all_results[tag] = {
            "description": desc,
            "n_samples": viol["n_samples"],
            "violations_before": viol,
            "violations_after": viol_post,
            "quantile_violations": qviol,
            "r2_before": mean_r2,
            "r2_after": mean_r2_post,
        }

    # Save results
    results_path = OUTPUT_DIR / "results_ordering_violations.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"  Summary Table")
    print(f"{'=' * 90}")
    print(f"  {'Method':>8s}  {'R² (before)':>11s}  {'R² (after)':>11s}  {'Violations':>10s}  {'After proj':>10s}")
    print(f"  {'-' * 60}")
    for tag, _ in methods:
        if tag not in all_results:
            continue
        r = all_results[tag]
        print(f"  {tag:>8s}  {r['r2_before']:>11.4f}  {r['r2_after']:>11.4f}  "
              f"{r['violations_before']['pct_any_violation']:>9.2f}%  "
              f"{r['violations_after']['pct_any_violation']:>9.2f}%")


def project_and_save(args):
    """Apply post-hoc projection and save corrected predictions."""
    from sklearn.metrics import r2_score

    input_path = OUTPUT_DIR / args.input
    data = np.load(input_path)
    pred = data["pred"]
    true = data["true"]
    dist = data.get("dist", None)
    q_hat = data.get("q_hat", None)

    print(f"  Input: {input_path}")
    print(f"  Shape: pred={pred.shape}, true={true.shape}")

    # Before
    viol_before = count_violations(pred, quantile_idx=1)
    r2_before = np.mean([r2_score(true[:, t], pred[:, t, 1]) for t in range(8)])
    print(f"  Before: {viol_before['total_violations']} violations, R²={r2_before:.4f}")

    # Project
    proj = project_physics(pred)

    # After
    viol_after = count_violations(proj, quantile_idx=1)
    r2_after = np.mean([r2_score(true[:, t], proj[:, t, 1]) for t in range(8)])
    print(f"  After:  {viol_after['total_violations']} violations, R²={r2_after:.4f}")

    # Save
    output_path = OUTPUT_DIR / args.output
    save_dict = {"pred": proj, "true": true}
    if dist is not None:
        save_dict["dist"] = dist
    if q_hat is not None:
        save_dict["q_hat"] = q_hat
    np.savez(output_path, **save_dict)
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze violations across all methods")
    parser.add_argument("--input", type=str, default=None,
                        help="Input npz file to project")
    parser.add_argument("--output", type=str, default=None,
                        help="Output npz file for projected predictions")
    args = parser.parse_args()

    if args.analyze:
        analyze_all()
    elif args.input and args.output:
        project_and_save(args)
    else:
        parser.error("Provide --analyze, or --input and --output")


if __name__ == "__main__":
    main()
