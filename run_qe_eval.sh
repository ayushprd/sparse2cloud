#!/bin/bash
# Evaluate quantile ensemble (G-QE) and re-run all analyses
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}

echo "=== Evaluating G-QE (quantile ensemble, 5 members) ==="
$PYTHON -u 53_evaluate_uncertainty.py --model-tag G-QE --mode quantile_ensemble \
    --ensemble-tags G-Q,G-Q2,G-Q3,G-Q4,G-Q5

echo "=== Updated comparison table ==="
$PYTHON -u 53_evaluate_uncertainty.py --compare

echo "=== Ordering violations for G-QE ==="
$PYTHON -u 57_posthoc_physics.py --analyze

echo "=== Selective prediction (including G-QE) ==="
$PYTHON -u 54_downstream_selective.py --methods G-QE,G-Q,G-QP,G-QPC,G-HET,G-EVI,G-MCD,G-DE

echo "=== Cloud classification with G-QE ==="
$PYTHON -u 55_downstream_classification.py --pred-file cqr_test_data_G-QE.npz

echo "=== Regenerate ECCV figures ==="
$PYTHON -u 60_eccv_figures.py

echo "=== ALL EVALUATION DONE ==="
