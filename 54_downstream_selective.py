"""Selective prediction: rejection curves showing R² vs fraction retained.

Demonstrates that calibrated uncertainty enables effective selective prediction —
rejecting the most uncertain predictions significantly improves R².

Usage:
    python -u 54_downstream_selective.py --pred-dir outputs/ --methods G-QPC,G-Q,G-HET,G-MCD,G-DE
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pathlib import Path

from config import OUTPUT_DIR, FIGURE_DIR

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]

TARGET_LABELS = {
    "centroid": "Centroid", "cloud_top": "Cloud Top",
    "cloud_base": "Cloud Base", "peak_level": "Peak Level",
    "thickness": "Thickness", "core_iwc": "Core IWC",
    "mean_iwc": "Mean IWC", "log_iwp": "log(IWP)",
}


def compute_rejection_curve(pred, true, widths, fractions):
    """Compute R² at each retention fraction.

    Args:
        pred: (N,) predictions
        true: (N,) true values
        widths: (N,) uncertainty (interval width)
        fractions: list of retention fractions

    Returns:
        list of R² values
    """
    sort_idx = np.argsort(widths)  # ascending = most certain first
    results = []
    for frac in fractions:
        n_keep = max(10, int(len(pred) * frac))
        kept = sort_idx[:n_keep]
        if len(np.unique(true[kept])) < 2:
            results.append(float('nan'))
        else:
            results.append(float(r2_score(true[kept], pred[kept])))
    return results


def aurrc(r2_values, fractions):
    """Area Under R² Retention Curve (trapezoidal integration)."""
    valid = [(f, r) for f, r in zip(fractions, r2_values) if not np.isnan(r)]
    if len(valid) < 2:
        return 0.0
    fracs, r2s = zip(*valid)
    return float(np.trapz(r2s, fracs))


def load_predictions(method_tag, pred_dir):
    """Load predictions from npz file. Returns (pred, true, widths).

    pred: (N, 8) point estimates
    true: (N, 8) true values
    widths: (N, 8) interval widths
    """
    # Try CQR test data first
    path = pred_dir / f"cqr_test_data_{method_tag}.npz"
    if path.exists():
        data = np.load(path)
        pred_q = data["pred"]   # (N, 8, 3)
        true = data["true"]     # (N, 8)
        q_hat = data.get("q_hat", np.zeros(8))
        widths = (pred_q[:, :, 2] + q_hat) - (pred_q[:, :, 0] - q_hat)
        return pred_q[:, :, 1], true, widths

    # Try uncertainty predictions
    path = pred_dir / f"uncertainty_pred_{method_tag.lower().replace('-', '_')}.npz"
    if path.exists():
        data = np.load(path)
        pred_q = data["pred"]  # (N, 8, 3)
        true = data["true"]
        widths = pred_q[:, :, 2] - pred_q[:, :, 0]
        return pred_q[:, :, 1], true, widths

    raise FileNotFoundError(f"No predictions found for {method_tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated method tags")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    methods = args.methods.split(",")
    fractions = np.arange(1.0, 0.49, -0.05).tolist()

    print(f"Selective prediction analysis")
    print(f"Methods: {methods}")
    print(f"Retention fractions: {len(fractions)} values from 1.0 to 0.5")

    all_results = {}

    for method in methods:
        print(f"\n  Loading {method}...")
        try:
            pred, true, widths = load_predictions(method, pred_dir)
        except FileNotFoundError as e:
            print(f"    SKIPPED: {e}")
            continue

        print(f"    {len(pred):,} pixels")

        method_results = {"per_target": {}, "fractions": fractions}

        # Per-target rejection curves
        for t, name in enumerate(TARGET_NAMES):
            r2_curve = compute_rejection_curve(pred[:, t], true[:, t], widths[:, t], fractions)
            auc = aurrc(r2_curve, fractions)
            method_results["per_target"][name] = {
                "r2_curve": r2_curve,
                "aurrc": auc,
                "r2_full": r2_curve[0],
                "r2_80pct": r2_curve[4] if len(r2_curve) > 4 else None,
            }

        # Mean across targets
        mean_curves = []
        for t in range(len(TARGET_NAMES)):
            name = TARGET_NAMES[t]
            mean_curves.append(method_results["per_target"][name]["r2_curve"])
        mean_curve = np.nanmean(mean_curves, axis=0).tolist()
        method_results["mean_r2_curve"] = mean_curve
        method_results["mean_aurrc"] = aurrc(mean_curve, fractions)

        all_results[method] = method_results
        print(f"    Mean AURRC: {method_results['mean_aurrc']:.4f}")
        print(f"    R² @ 100%: {mean_curve[0]:.4f}, @ 80%: {mean_curve[4]:.4f}")

    # Save results
    results_path = OUTPUT_DIR / "results_selective_prediction.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Mean rejection curves
    ax = axes[0]
    for method in methods:
        if method not in all_results:
            continue
        r = all_results[method]
        label = f"{method} (AURRC={r['mean_aurrc']:.3f})"
        ax.plot(fractions, r["mean_r2_curve"], "o-", markersize=3, label=label)
    ax.set_xlabel("Fraction retained")
    ax.set_ylabel("Mean R²")
    ax.set_title("(a) Selective Prediction — Mean Across Targets")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.48, 1.02)

    # (b) Per-target for best method
    ax = axes[1]
    best_method = max((m for m in methods if m in all_results),
                      key=lambda m: all_results[m]["mean_aurrc"], default=None)
    if best_method:
        r = all_results[best_method]
        for name in TARGET_NAMES[:6]:
            curve = r["per_target"][name]["r2_curve"]
            ax.plot(fractions, curve, "o-", markersize=3, label=TARGET_LABELS[name])
        ax.set_xlabel("Fraction retained")
        ax.set_ylabel("R²")
        ax.set_title(f"(b) Per-Target — {best_method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.48, 1.02)

    plt.tight_layout()
    fig_path = FIGURE_DIR / "selective_prediction.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")
    print("Done!")


if __name__ == "__main__":
    main()
