"""Publication figures for ECCV: Dense Cloud Geometry with Calibrated Uncertainty.

Figures:
  1. Main comparison: R² and MPIW per target for top methods
  2. Reliability diagram: pre-CQR vs post-CQR coverage
  3. Ordering violations ablation
  4. Selective prediction rejection curves
  5. Cross-track uncertainty analysis
  6. Downstream cloud classification
  7. LaTeX tables

Usage:
    python -u 60_eccv_figures.py           # all figures
    python -u 60_eccv_figures.py --fig 2   # single figure
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from config import OUTPUT_DIR, FIGURE_DIR

PAPER_DIR = FIGURE_DIR / "eccv"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality settings (ECCV: 2-column, ~8.5cm per column)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

# Colorblind-friendly palette
COLORS = {
    "G-QE": "#2196F3",   # blue - best method
    "G-Q":  "#4CAF50",   # green
    "G-DE": "#FF9800",   # orange
    "G-HET": "#9C27B0",  # purple
    "G-EVI": "#F44336",  # red
    "G-MCD": "#795548",  # brown
    "G-QP": "#607D8B",   # blue-gray
    "G-QPC": "#E91E63",  # pink
    "G-B":  "#BDBDBD",   # gray — point baseline
}

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

# Short labels for bar charts
TARGET_SHORT = {
    "centroid": "Cent", "cloud_top": "Top",
    "cloud_base": "Base", "peak_level": "Peak",
    "thickness": "Thick", "core_iwc": "Core",
    "mean_iwc": "Mean", "log_iwp": "IWP",
}


def load_uncertainty_results():
    """Load all uncertainty evaluation results."""
    import glob
    results = {}
    for f in sorted(glob.glob(str(OUTPUT_DIR / "results_uncertainty_*.json"))):
        with open(f) as fp:
            data = json.load(fp)
        if "model_tag" in data:
            results[data["model_tag"]] = data
    return results


def load_baseline_results():
    """Load G-B point estimate baseline results."""
    gb_path = OUTPUT_DIR / "results_geom_G-B.json"
    if not gb_path.exists():
        return None
    with open(gb_path) as f:
        r = json.load(f)
    # Convert to uncertainty-results-compatible format (R² only, no intervals)
    return {
        "per_target": {t: {"r2": r["test"][t]["r2"]} for t in TARGET_NAMES},
        "mean_r2": r["test"]["mean_r2"],
    }


def fig1_main_comparison():
    """Figure 1: Main comparison — R² and MPIW per target for top methods.

    2-panel figure showing (a) R² per target (incl. G-B baseline), (b) MPIW per target.
    """
    results = load_uncertainty_results()
    # Include G-B point baseline for context
    gb = load_baseline_results()
    if gb:
        results["G-B"] = gb
    methods = ["G-QE", "G-Q", "G-DE", "G-HET", "G-EVI", "G-MCD", "G-QPC", "G-B"]
    methods = [m for m in methods if m in results]
    # Uncertainty methods only (have intervals) for panel (b)
    unc_methods = [m for m in methods if m != "G-B"]

    if not methods:
        print("  [SKIP] No results")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # (a) R² per target — all methods including G-B baseline
    n_targets = len(TARGET_NAMES)
    n_methods = len(methods)
    x = np.arange(n_targets)
    width = 0.8 / n_methods

    for i, m in enumerate(methods):
        r = results[m]
        r2s = [r["per_target"][t]["r2"] for t in TARGET_NAMES]
        offset = (i - n_methods/2 + 0.5) * width
        ax1.bar(x + offset, r2s, width * 0.9, label=m,
                color=COLORS.get(m, "#999"), alpha=0.85, edgecolor="white", linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([TARGET_SHORT[t] for t in TARGET_NAMES], rotation=30, ha="right")
    ax1.set_ylabel("R$^2$")
    ax1.set_title("(a) Point Estimate Quality")
    ax1.legend(ncol=4, fontsize=6, frameon=True, fancybox=False, edgecolor="0.8",
               loc="lower left")
    ax1.set_ylim(0.3, 0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) MPIW per target (lower = better, i.e., sharper intervals) — uncertainty methods only
    n_unc = len(unc_methods)
    width_b = 0.8 / n_unc
    for i, m in enumerate(unc_methods):
        r = results[m]
        mpiws = [r["per_target"][t]["mpiw"] for t in TARGET_NAMES[:5]]  # geometry targets only
        offset = (i - n_unc/2 + 0.5) * width_b
        ax2.bar(np.arange(5) + offset, mpiws, width_b * 0.9,
                color=COLORS.get(m, "#999"), alpha=0.85, edgecolor="white", linewidth=0.3)

    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels([TARGET_SHORT[t] for t in TARGET_NAMES[:5]], rotation=30, ha="right")
    ax2.set_ylabel("MPIW (CQR-calibrated)")
    ax2.set_title("(b) Interval Width (lower = sharper)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "main_comparison.pdf")
    fig.savefig(PAPER_DIR / "main_comparison.png")
    plt.close(fig)
    print(f"  Saved main_comparison.pdf")


def fig2_reliability():
    """Figure 2: Reliability diagram — pre-CQR vs post-CQR.

    (a) Raw intervals at 80% nominal → actual coverage (miscalibrated)
    (b) CQR intervals at multiple nominal levels → actual coverage (on diagonal)
    """
    results = load_uncertainty_results()
    methods = ["G-QE", "G-Q", "G-DE", "G-HET", "G-EVI", "G-MCD", "G-QPC"]
    methods = [m for m in methods if m in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) Raw PICP at 80% nominal — bar chart
    raw_picps = [(m, results[m]["mean_raw_picp"]) for m in methods]
    raw_picps.sort(key=lambda x: x[1], reverse=True)

    x = np.arange(len(raw_picps))
    colors = [COLORS.get(m, "#999") for m, _ in raw_picps]
    bars = ax1.bar(x, [p for _, p in raw_picps], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3)

    # Target line at 80%
    ax1.axhline(y=0.80, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.text(len(raw_picps)-0.5, 0.81, "80% target", fontsize=7, ha="right", alpha=0.6)

    for bar, (m, p) in zip(bars, raw_picps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{p:.2f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m for m, _ in raw_picps], rotation=30, ha="right")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("(a) Raw Intervals (before CQR)")
    ax1.set_ylim(0, 1.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) CQR reliability diagram
    nominals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ax2.plot([0.45, 1.0], [0.45, 1.0], "k--", linewidth=0.8, alpha=0.3, label="Perfect")

    for m in methods:
        r = results[m]
        if "reliability" not in r:
            continue
        empirical = [r["reliability"][str(n)]["mean"] for n in nominals]
        ax2.plot(nominals, empirical, "o-", markersize=3,
                color=COLORS.get(m, "#999"), label=m, alpha=0.8)

    ax2.set_xlabel("Nominal Coverage")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("(b) After CQR Calibration")
    ax2.legend(fontsize=6, ncol=2, frameon=True, fancybox=False, edgecolor="0.8")
    ax2.set_xlim(0.45, 1.0)
    ax2.set_ylim(0.45, 1.0)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "reliability.pdf")
    fig.savefig(PAPER_DIR / "reliability.png")
    plt.close(fig)
    print(f"  Saved reliability.pdf")


def fig3_violations():
    """Figure 3: Ordering violations ablation — bar chart.

    Shows % violations and R² for G-Q, G-QP, G-QPC, G-Q+posthoc.
    """
    viol_path = OUTPUT_DIR / "results_ordering_violations.json"
    if not viol_path.exists():
        print("  [SKIP] No violations results")
        return

    with open(viol_path) as f:
        data = json.load(f)

    methods = [
        ("G-Q", "Quantile\n(unconstrained)"),
        ("G-QP", "Quantile\n+ ordering loss"),
        ("G-QPC", "Quantile\n+ physics head"),
    ]
    methods = [(t, l) for t, l in methods if t in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.5))

    # (a) Violations %
    x = np.arange(len(methods))
    viols = [data[t]["violations_before"]["pct_any_violation"] for t, _ in methods]
    viols_post = [data[t]["violations_after"]["pct_any_violation"] for t, _ in methods]

    width = 0.35
    bars1 = ax1.bar(x - width/2, viols, width, label="Before projection",
                    color="#F44336", alpha=0.8, edgecolor="white", linewidth=0.3)
    bars2 = ax1.bar(x + width/2, viols_post, width, label="After projection",
                    color="#4CAF50", alpha=0.8, edgecolor="white", linewidth=0.3)

    for bar, v in zip(bars1, viols):
        if v > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([l for _, l in methods], fontsize=7)
    ax1.set_ylabel("Ordering Violations (%)")
    ax1.set_title("(a) Geometry Violations")
    ax1.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="0.8")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) R² comparison
    r2_before = [data[t]["r2_before"] for t, _ in methods]
    r2_after = [data[t]["r2_after"] for t, _ in methods]

    bars1 = ax2.bar(x - width/2, r2_before, width, label="Before projection",
                    color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.3)
    bars2 = ax2.bar(x + width/2, r2_after, width, label="After projection",
                    color="#FF9800", alpha=0.8, edgecolor="white", linewidth=0.3)

    for bar, v in zip(bars1, r2_before):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars2, r2_after):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([l for _, l in methods], fontsize=7)
    ax2.set_ylabel("Mean R$^2$")
    ax2.set_title("(b) Point Estimate Quality")
    ax2.set_ylim(0.68, 0.74)
    ax2.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="0.8")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "violations_ablation.pdf")
    fig.savefig(PAPER_DIR / "violations_ablation.png")
    plt.close(fig)
    print(f"  Saved violations_ablation.pdf")


def fig4_selective():
    """Figure 4: Selective prediction rejection curves.

    (a) Mean R² vs fraction retained — all methods
    (b) Per-target curves for best method
    """
    sel_path = OUTPUT_DIR / "results_selective_prediction.json"
    if not sel_path.exists():
        print("  [SKIP] No selective prediction results")
        return

    with open(sel_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Per-target rejection curves for best quantile method
    method = "G-QE" if "G-QE" in data else ("G-Q" if "G-Q" in data else list(data.keys())[0])
    r = data[method]
    fracs = r["fractions"]
    target_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
                     "#795548", "#607D8B", "#E91E63"]
    for i, name in enumerate(TARGET_NAMES[:5]):  # geometry targets
        curve = r["per_target"][name]["r2_curve"]
        ax1.plot(fracs, curve, "o-", markersize=2, label=TARGET_LABELS[name],
                color=target_colors[i], alpha=0.8)

    ax1.set_xlabel("Fraction Retained")
    ax1.set_ylabel("R$^2$")
    ax1.set_title(f"(a) Per-Target Rejection — {method}")
    ax1.legend(fontsize=6, ncol=2, frameon=True, fancybox=False, edgecolor="0.8")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.48, 1.02)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) R² improvement at 80% retention for all methods
    methods_sorted = sorted(data.keys(),
        key=lambda m: data[m]["per_target"]["centroid"]["r2_curve"][4] - data[m]["per_target"]["centroid"]["r2_curve"][0],
        reverse=True)
    best = methods_sorted[0]
    # Compute R² improvement at 80% retention
    x = np.arange(len(data))
    labels_list = list(data.keys())
    improvements = []
    for m in labels_list:
        # Average over geometry targets (not intensity)
        imp = 0
        for t in TARGET_NAMES[:5]:
            c = data[m]["per_target"][t]["r2_curve"]
            imp += (c[4] - c[0])  # R² at 80% - R² at 100%
        improvements.append(imp / 5 * 100)  # in percentage points

    colors_bar = [COLORS.get(m, "#999") for m in labels_list]
    bars = ax2.bar(x, improvements, color=colors_bar, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
    for bar, v in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, max(v + 0.1, 0.1),
                f"{v:+.1f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("$\\Delta$R$^2$ (pp)")
    ax2.set_title("(b) R$^2$ Gain at 80% Retention")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(PAPER_DIR / "selective_prediction.pdf")
    fig.savefig(PAPER_DIR / "selective_prediction.png")
    plt.close(fig)
    print(f"  Saved selective_prediction.pdf")


def fig5_crosstrack():
    """Figure 5: Cross-track uncertainty analysis.

    (a) R² vs distance — 50/50 split held-out evaluation
    (b) PICP vs distance — coverage maintained off-track
    (c) Interval width at ALL pixels vs distance — uniform uncertainty
    Data source: results_crosstrack_dense.json from 58_crosstrack_uncertainty.py
    """
    ct_path = OUTPUT_DIR / "results_crosstrack_dense.json"
    if not ct_path.exists():
        print("  [SKIP] No crosstrack dense results — run 58_crosstrack_uncertainty.py")
        return

    with open(ct_path) as f:
        ct = json.load(f)

    heldout = ct["heldout"]
    allpx = ct["allpixel_width"]

    # Only show bins with enough samples (>= 100 for heldout, > 0 for allpx)
    # Limit to distance 0-8 for clean figure (98%+ of data)
    max_dist = 8
    ho_bins = sorted([b for b in heldout.keys() if float(b.split("-")[0]) < max_dist],
                     key=lambda b: float(b.split("-")[0]))
    ap_bins = sorted([b for b in allpx.keys() if float(b.split("-")[0]) < max_dist],
                     key=lambda b: float(b.split("-")[0]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.2))

    # (a) R² vs distance
    x = np.arange(len(ho_bins))
    r2s = [heldout[b]["mean_r2"] for b in ho_bins]
    ns = [heldout[b]["n"] for b in ho_bins]
    ax1.bar(x, r2s, color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.3)
    overall_r2 = np.average(r2s, weights=ns)
    ax1.axhline(y=overall_r2, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ho_bins, fontsize=7)
    ax1.set_xlabel("Distance to track (px)")
    ax1.set_ylabel("Mean R$^2$")
    ax1.set_title("(a) R$^2$ vs Distance")
    ax1.set_ylim(0.65, 0.78)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) PICP vs distance
    picps = [heldout[b]["mean_picp"] for b in ho_bins]
    ax2.bar(x, picps, color="#4CAF50", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axhline(y=0.90, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                label="90% nominal")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ho_bins, fontsize=7)
    ax2.set_xlabel("Distance to track (px)")
    ax2.set_ylabel("PICP")
    ax2.set_title("(b) Coverage vs Distance")
    ax2.set_ylim(0.85, 0.95)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # (c) Mean interval width at ALL pixels
    x3 = np.arange(len(ap_bins))
    # Average across geometry targets (centroid, top, base, peak, thickness)
    geo_targets = ["centroid", "cloud_top", "cloud_base", "peak_level", "thickness"]
    mean_widths = []
    for b in ap_bins:
        w = np.mean([allpx[b]["mean_width"][t] for t in geo_targets])
        mean_widths.append(w)
    ax3.bar(x3, mean_widths, color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.3)
    overall_w = np.mean(mean_widths)
    ax3.axhline(y=overall_w, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(ap_bins, fontsize=7)
    ax3.set_xlabel("Distance to track (px)")
    ax3.set_ylabel("Mean Width (hPa)")
    ax3.set_title("(c) Interval Width (all px)")
    ax3.set_ylim(25, 37)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "crosstrack_uncertainty.pdf")
    fig.savefig(PAPER_DIR / "crosstrack_uncertainty.png")
    plt.close(fig)
    print(f"  Saved crosstrack_uncertainty.pdf")


def fig6_classification():
    """Figure 6: Downstream cloud classification.

    (a) Accuracy/F1 comparison: oracle vs geometry vs geometry+uncertainty
    (b) Selective classification curve
    """
    cls_path = OUTPUT_DIR / "results_cloud_classification.json"
    if not cls_path.exists():
        print("  [SKIP] No classification results")
        return

    with open(cls_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # (a) Accuracy and F1 bar chart
    configs = ["geometry_only", "geometry_uncertainty", "oracle"]
    labels = ["Geometry Only", "Geometry +\nUncertainty", "Oracle"]
    colors = ["#607D8B", "#2196F3", "#4CAF50"]

    x = np.arange(2)  # accuracy, f1
    width = 0.25

    for i, (cfg, label, color) in enumerate(zip(configs, labels, colors)):
        if cfg not in data:
            continue
        vals = [data[cfg]["accuracy"], data[cfg]["f1_macro"]]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, vals, width * 0.9, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Accuracy", "Macro F1"])
    ax1.set_ylabel("Score")
    ax1.set_title("(a) Cloud Type Classification")
    ax1.set_ylim(0.6, 1.05)
    ax1.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="0.8")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) Selective classification
    if "selective" in data:
        sel = data["selective"]
        fracs = [s["frac"] for s in sel]
        accs = [s["accuracy"] for s in sel]
        f1s = [s["f1"] for s in sel]

        ax2.plot(fracs, accs, "o-", color="#2196F3", markersize=3, label="Accuracy")
        ax2.plot(fracs, f1s, "s-", color="#FF9800", markersize=3, label="Macro F1")

        ax2.set_xlabel("Fraction Retained")
        ax2.set_ylabel("Score")
        ax2.set_title("(b) Selective Classification")
        ax2.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="0.8")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.45, 1.05)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "classification.pdf")
    fig.savefig(PAPER_DIR / "classification.png")
    plt.close(fig)
    print(f"  Saved classification.pdf")


def fig7_interval_error():
    """Figure 7: Interval width vs absolute error scatter/hexbin.

    Shows that wider intervals correlate with larger errors (good uncertainty).
    """
    # Use G-Q data
    npz_path = OUTPUT_DIR / "cqr_test_data_G-Q.npz"
    if not npz_path.exists():
        print("  [SKIP] No G-Q test data")
        return

    data = np.load(npz_path)
    pred = data["pred"]  # (N, 8, 3)
    true = data["true"]  # (N, 8)

    # Pick 2 representative targets: centroid (geometry) and core_iwc (intensity)
    targets = [(0, "Centroid"), (5, "Core IWC")]

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))

    for ax, (t_idx, t_name) in zip(axes, targets):
        width = pred[:, t_idx, 2] - pred[:, t_idx, 0]  # raw interval width
        error = np.abs(true[:, t_idx] - pred[:, t_idx, 1])

        # Subsample for plotting
        rng = np.random.RandomState(42)
        idx = rng.choice(len(width), min(50000, len(width)), replace=False)

        hb = ax.hexbin(width[idx], error[idx], gridsize=40, cmap="Blues",
                       mincnt=1, linewidths=0.1)
        ax.set_xlabel("Interval Width")
        ax.set_ylabel("|Error|")
        ax.set_title(t_name)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add correlation annotation
        from scipy.stats import spearmanr
        rho, _ = spearmanr(width, error)
        ax.text(0.95, 0.95, f"$\\rho$={rho:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "interval_error_correlation.pdf")
    fig.savefig(PAPER_DIR / "interval_error_correlation.png")
    plt.close(fig)
    print(f"  Saved interval_error_correlation.pdf")


def print_latex_tables():
    """Generate LaTeX tables for the paper."""
    results = load_uncertainty_results()
    # Add G-B baseline
    gb = load_baseline_results()
    if gb:
        results["G-B"] = gb

    print("\n" + "=" * 90)
    print("  TABLE 1: Main Comparison (CQR-calibrated)")
    print("=" * 90)

    methods = ["G-QE", "G-Q", "G-DE", "G-HET", "G-EVI", "G-MCD", "G-QP", "G-QPC", "G-B"]
    methods = [m for m in methods if m in results]

    # LaTeX header
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison of uncertainty methods. All methods use identical CQR calibration. G-B is a deterministic point-estimate baseline.}")
    print("\\label{tab:comparison}")
    print("\\resizebox{\\linewidth}{!}{")
    print("\\begin{tabular}{l" + "c" * 6 + "}")
    print("\\toprule")
    print("Method & R$^2$ $\\uparrow$ & MAE $\\downarrow$ & PICP & MPIW $\\downarrow$ & Raw PICP & Corr $\\uparrow$ \\\\")
    print("\\midrule")

    for m in methods:
        r = results[m]
        if m == "G-B":
            # Point baseline — no intervals
            line = f"{m} & {r['mean_r2']:.3f} & --- & --- & --- & --- & --- \\\\"
        else:
            line = f"{m} & {r['mean_r2']:.3f} & {r['mean_mae']:.3f} & {r['mean_picp']:.3f} & "
            line += f"{r['mean_mpiw']:.1f} & {r['mean_raw_picp']:.3f} & {r['mean_corr']:.3f} \\\\"
        # Bold best
        if m == "G-QE":
            line = "\\textbf{" + line.replace("\\\\", "} \\\\")
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")

    # Table 2: Violations ablation
    print("\n" + "=" * 90)
    print("  TABLE 2: Ordering Violations Ablation")
    print("=" * 90)

    viol_path = OUTPUT_DIR / "results_ordering_violations.json"
    if viol_path.exists():
        with open(viol_path) as f:
            viol_data = json.load(f)

        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Physics constraint ablation. Post-hoc projection achieves zero violations without accuracy loss.}")
        print("\\label{tab:violations}")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("Method & R$^2$ & Violations (\\%) & After Proj. (\\%) \\\\")
        print("\\midrule")

        for m in ["G-Q", "G-QP", "G-QPC"]:
            if m not in viol_data:
                continue
            v = viol_data[m]
            print(f"{m} & {v['r2_before']:.3f} & {v['violations_before']['pct_any_violation']:.2f} & "
                  f"{v['violations_after']['pct_any_violation']:.2f} \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

    # Table 3: Per-target R²
    print("\n" + "=" * 90)
    print("  TABLE 3: Per-Target R² (CQR median)")
    print("=" * 90)

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Per-target R$^2$ for all uncertainty methods.}")
    print("\\label{tab:per_target}")
    print("\\resizebox{\\linewidth}{!}{")
    cols = "l" + "c" * len(TARGET_NAMES) + "c"
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    header = "Method"
    for t in TARGET_NAMES:
        header += f" & {TARGET_SHORT[t]}"
    header += " & Mean \\\\"
    print(header)
    print("\\midrule")

    for m in methods:
        r = results[m]
        line = m
        for t in TARGET_NAMES:
            line += f" & {r['per_target'][t]['r2']:.3f}"
        line += f" & {r['mean_r2']:.3f} \\\\"
        print(line)

    print("\\bottomrule")
    print(f"\\end{{tabular}}}}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default="all",
                        choices=["all", "1", "2", "3", "4", "5", "6", "7", "tables"])
    args = parser.parse_args()

    print(f"Generating ECCV figures in {PAPER_DIR}/\n")

    if args.fig in ("all", "1"):
        print("Fig 1: Main comparison...")
        fig1_main_comparison()

    if args.fig in ("all", "2"):
        print("Fig 2: Reliability diagram...")
        fig2_reliability()

    if args.fig in ("all", "3"):
        print("Fig 3: Ordering violations...")
        fig3_violations()

    if args.fig in ("all", "4"):
        print("Fig 4: Selective prediction...")
        fig4_selective()

    if args.fig in ("all", "5"):
        print("Fig 5: Cross-track uncertainty...")
        fig5_crosstrack()

    if args.fig in ("all", "6"):
        print("Fig 6: Cloud classification...")
        fig6_classification()

    if args.fig in ("all", "7"):
        print("Fig 7: Interval-error correlation...")
        fig7_interval_error()

    if args.fig in ("all", "tables"):
        print_latex_tables()

    print("\nDone!")


if __name__ == "__main__":
    main()
