#!/bin/bash
# Run all remaining experiments sequentially
# Priority: G-QPC > G-HET > G-DE2-5 > G-EVI > G-PC

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}

echo "=== Starting G-QPC (quantile + physics head) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-QPC --mode quantile --physics-head --seed 42 --epochs 100 --patience 20

echo "=== Starting G-PC (point + physics head) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-PC --mode point --physics-head --seed 42 --epochs 100 --patience 20

echo "=== Starting G-HET (heteroscedastic) ==="
$PYTHON -u 52_train_uncertainty_baselines.py --tag G-HET --mode heteroscedastic --seed 42 --epochs 100 --patience 20

echo "=== Starting G-EVI (evidential) ==="
$PYTHON -u 52_train_uncertainty_baselines.py --tag G-EVI --mode evidential --seed 42 --epochs 100 --patience 20

echo "=== Starting G-DE2 (ensemble seed 123) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-DE2 --mode point --seed 123 --epochs 100 --patience 20

echo "=== Starting G-DE3 (ensemble seed 777) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-DE3 --mode point --seed 777 --epochs 100 --patience 20

# G-DE4 and G-DE5 skipped to save time (3-member ensemble sufficient)
# echo "=== Starting G-DE4 (ensemble seed 2024) ==="
# $PYTHON -u 47_train_geometry_quantile.py --tag G-DE4 --mode point --seed 2024 --epochs 100 --patience 20
# echo "=== Starting G-DE5 (ensemble seed 31415) ==="
# $PYTHON -u 47_train_geometry_quantile.py --tag G-DE5 --mode point --seed 31415 --epochs 100 --patience 20

echo "=== ALL TRAINING DONE ==="
echo "Now running evaluations..."

# MC-Dropout eval (no training)
echo "=== MC-Dropout evaluation ==="
$PYTHON -u 53_evaluate_uncertainty.py --model-tag G-MCD --mode mc_dropout --base-model geom_G-B.pt --T 30

# Unified evaluation for all trained methods
for TAG in G-B G-Q G-QP G-QPC G-PC; do
    echo "=== Evaluating $TAG ==="
    MODE="quantile"
    if [ "$TAG" = "G-B" ] || [ "$TAG" = "G-PC" ]; then
        # Skip point-only models for uncertainty eval (no intervals)
        continue
    fi
    if [ "$TAG" = "G-QPC" ]; then
        MODE="physics_quantile"
    fi
    $PYTHON -u 53_evaluate_uncertainty.py --model-tag $TAG --mode $MODE
done

echo "=== Evaluating G-HET ==="
$PYTHON -u 53_evaluate_uncertainty.py --model-tag G-HET --mode heteroscedastic

echo "=== Evaluating G-EVI ==="
$PYTHON -u 53_evaluate_uncertainty.py --model-tag G-EVI --mode evidential

echo "=== Evaluating G-DE (ensemble) ==="
$PYTHON -u 53_evaluate_uncertainty.py --model-tag G-DE --mode ensemble --ensemble-tags G-B,G-DE2,G-DE3

# Comparison
echo "=== Comparison table ==="
$PYTHON -u 53_evaluate_uncertainty.py --compare

# Downstream tasks (use G-QPC as primary)
echo "=== Downstream: Selective prediction ==="
$PYTHON -u 54_downstream_selective.py --methods G-Q,G-QP,G-QPC,G-HET,G-EVI,G-MCD,G-DE

echo "=== Downstream: Cloud classification ==="
$PYTHON -u 55_downstream_classification.py --pred-file cqr_test_data_G-QPC.npz

echo "=== ALL DONE ==="
