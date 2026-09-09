"""Uncertainty-aware cloud type classification downstream task.

Defines 4 cloud types from geometry targets and trains simple classifiers
with and without uncertainty features to show calibrated uncertainty improves
downstream task performance.

Usage:
    python -u 55_downstream_classification.py --pred-file cqr_test_data_G-QPC.npz
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             brier_score_loss)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import OUTPUT_DIR, FIGURE_DIR

TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]

CLOUD_TYPES = {
    0: "Thin Cirrus",
    1: "Thick Ice",
    2: "Deep Conv.",
    3: "Mid-level",
}


def classify_cloud_type(geometry):
    """Assign cloud type from geometry targets.

    Args:
        geometry: (N, 8) [centroid, cloud_top, cloud_base, peak_level,
                          thickness, core_iwc, mean_iwc, log_iwp]

    Returns:
        (N,) integer labels 0-3
    """
    cloud_top = geometry[:, 1]
    thickness = geometry[:, 4]
    core_iwc = geometry[:, 5]

    labels = np.full(len(geometry), 3, dtype=np.int64)  # default: mid-level

    # Note: levels are inverted (0=TOA, 158=surface)
    # High cloud: cloud_top > 100 means high in atmosphere (closer to TOA)
    high = cloud_top > 100

    # Thin cirrus: high, thin, low IWC
    thin_cirrus = high & (thickness < 20) & (core_iwc < -3.0)
    labels[thin_cirrus] = 0

    # Thick ice: high, thicker, moderate IWC
    thick_ice = high & (thickness >= 20) & (thickness < 50)
    labels[thick_ice] = 1

    # Deep convection: very thick (regardless of height)
    deep_conv = thickness >= 50
    labels[deep_conv] = 2

    # Mid-level: everything else (low cloud_top OR medium that doesn't fit)
    # Already set as default

    return labels


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error for multi-class predictions."""
    n_classes = y_prob.shape[1]
    ece = 0.0
    n_total = len(y_true)

    for c in range(n_classes):
        probs = y_prob[:, c]
        true_binary = (y_true == c).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            avg_confidence = probs[mask].mean()
            avg_accuracy = true_binary[mask].mean()
            ece += mask.sum() / n_total * abs(avg_accuracy - avg_confidence)

    return float(ece / n_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True,
                        help="NPZ file with predictions (in outputs/)")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of test data to use for classifier training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load predictions
    pred_path = OUTPUT_DIR / args.pred_file
    print(f"Loading predictions: {pred_path}")
    data = np.load(pred_path)
    pred = data["pred"]    # (N, 8, 3) quantile predictions
    true = data["true"]    # (N, 8) true values
    q_hat = data.get("q_hat", np.zeros(8))

    N = len(true)
    print(f"  {N:,} test pixels")

    # Compute cloud type labels from TRUE geometry
    true_labels = classify_cloud_type(true)

    # Report class distribution
    print(f"\n  Cloud type distribution (from true geometry):")
    for c, name in CLOUD_TYPES.items():
        n = (true_labels == c).sum()
        print(f"    {c} ({name:>12s}): {n:,} ({n/N*100:.1f}%)")

    # Point predictions and uncertainty
    pred_mean = pred[:, :, 1]  # median
    widths = (pred[:, :, 2] + q_hat) - (pred[:, :, 0] - q_hat)

    # Split into train/test for classifier
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(N)
    n_train = int(N * args.train_frac)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    results = {}

    # Experiment 1: Oracle (true geometry → cloud type)
    print(f"\n  Training classifiers ({n_train:,} train, {len(test_idx):,} test)...")
    scaler_oracle = StandardScaler()
    X_train = scaler_oracle.fit_transform(true[train_idx])
    X_test = scaler_oracle.transform(true[test_idx])
    y_train = true_labels[train_idx]
    y_test = true_labels[test_idx]

    clf_oracle = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                                random_state=args.seed)
    clf_oracle.fit(X_train, y_train)
    y_pred_oracle = clf_oracle.predict(X_test)
    y_prob_oracle = clf_oracle.predict_proba(X_test)

    acc_oracle = accuracy_score(y_test, y_pred_oracle)
    f1_oracle = f1_score(y_test, y_pred_oracle, average="macro")
    ece_oracle = expected_calibration_error(y_test, y_prob_oracle)
    results["oracle"] = {"accuracy": acc_oracle, "f1_macro": f1_oracle, "ece": ece_oracle}
    print(f"    Oracle:        acc={acc_oracle:.4f}, F1={f1_oracle:.4f}, ECE={ece_oracle:.4f}")

    # Experiment 2: Geometry-only (predicted geometry → cloud type)
    scaler_geo = StandardScaler()
    X_train_geo = scaler_geo.fit_transform(pred_mean[train_idx])
    X_test_geo = scaler_geo.transform(pred_mean[test_idx])

    clf_geo = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                             random_state=args.seed)
    clf_geo.fit(X_train_geo, y_train)
    y_pred_geo = clf_geo.predict(X_test_geo)
    y_prob_geo = clf_geo.predict_proba(X_test_geo)

    acc_geo = accuracy_score(y_test, y_pred_geo)
    f1_geo = f1_score(y_test, y_pred_geo, average="macro")
    ece_geo = expected_calibration_error(y_test, y_prob_geo)
    results["geometry_only"] = {"accuracy": acc_geo, "f1_macro": f1_geo, "ece": ece_geo}
    print(f"    Geometry-only: acc={acc_geo:.4f}, F1={f1_geo:.4f}, ECE={ece_geo:.4f}")

    # Experiment 3: Geometry + Uncertainty features
    X_train_unc = np.concatenate([pred_mean[train_idx], widths[train_idx]], axis=1)
    X_test_unc = np.concatenate([pred_mean[test_idx], widths[test_idx]], axis=1)
    scaler_unc = StandardScaler()
    X_train_unc = scaler_unc.fit_transform(X_train_unc)
    X_test_unc = scaler_unc.transform(X_test_unc)

    clf_unc = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                             random_state=args.seed)
    clf_unc.fit(X_train_unc, y_train)
    y_pred_unc = clf_unc.predict(X_test_unc)
    y_prob_unc = clf_unc.predict_proba(X_test_unc)

    acc_unc = accuracy_score(y_test, y_pred_unc)
    f1_unc = f1_score(y_test, y_pred_unc, average="macro")
    ece_unc = expected_calibration_error(y_test, y_prob_unc)
    results["geometry_uncertainty"] = {"accuracy": acc_unc, "f1_macro": f1_unc, "ece": ece_unc}
    print(f"    Geo+Uncert:    acc={acc_unc:.4f}, F1={f1_unc:.4f}, ECE={ece_unc:.4f}")

    # Experiment 4: Selective classification — reject by max uncertainty
    max_width = widths[test_idx].max(axis=1)  # max across targets per pixel
    sort_idx = np.argsort(max_width)  # ascending = most certain first

    fractions = np.arange(1.0, 0.49, -0.1).tolist()
    selective_results = []
    for frac in fractions:
        n_keep = max(10, int(len(test_idx) * frac))
        kept = sort_idx[:n_keep]
        acc = accuracy_score(y_test[kept], y_pred_geo[kept])
        f1 = f1_score(y_test[kept], y_pred_geo[kept], average="macro")
        selective_results.append({"frac": frac, "accuracy": acc, "f1": f1})
    results["selective"] = selective_results

    print(f"\n  Selective classification:")
    for sr in selective_results:
        print(f"    {sr['frac']*100:5.0f}% retained: acc={sr['accuracy']:.4f}, F1={sr['f1']:.4f}")

    # Save
    results_path = OUTPUT_DIR / f"results_cloud_classification.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Confusion matrices
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred_unc, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLOUD_TYPES)):
        for j in range(len(CLOUD_TYPES)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black", fontsize=9)
    labels = [CLOUD_TYPES[i] for i in range(len(CLOUD_TYPES))]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("(a) Confusion Matrix (Geo+Uncertainty)")

    # (b) Accuracy comparison bar chart
    ax = axes[1]
    methods = ["Oracle", "Geo-only", "Geo+Uncert"]
    accs = [acc_oracle, acc_geo, acc_unc]
    f1s = [f1_oracle, f1_geo, f1_unc]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color="tab:blue", alpha=0.8)
    ax.bar(x + w/2, f1s, w, label="Macro F1", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("(b) Classification Performance")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")

    # (c) Selective classification
    ax = axes[2]
    sel_fracs = [sr["frac"] for sr in selective_results]
    sel_accs = [sr["accuracy"] for sr in selective_results]
    sel_f1s = [sr["f1"] for sr in selective_results]
    ax.plot(sel_fracs, sel_accs, "o-", label="Accuracy", markersize=5)
    ax.plot(sel_fracs, sel_f1s, "s-", label="Macro F1", markersize=5)
    ax.set_xlabel("Fraction retained")
    ax.set_ylabel("Score")
    ax.set_title("(c) Selective Classification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.45, 1.05)

    plt.tight_layout()
    fig_path = FIGURE_DIR / "cloud_classification.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")
    print("Done!")


if __name__ == "__main__":
    main()
