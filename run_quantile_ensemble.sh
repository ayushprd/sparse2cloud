#!/bin/bash
# Train remaining quantile ensemble members (G-Q3, G-Q4, G-Q5) sequentially
# G-Q2 is already training separately
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}

echo "=== Starting G-Q3 (seed 777) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-Q3 --mode quantile --seed 777 --epochs 100 --patience 20

echo "=== Starting G-Q4 (seed 2024) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-Q4 --mode quantile --seed 2024 --epochs 100 --patience 20

echo "=== Starting G-Q5 (seed 31415) ==="
$PYTHON -u 47_train_geometry_quantile.py --tag G-Q5 --mode quantile --seed 31415 --epochs 100 --patience 20

echo "=== ALL QUANTILE ENSEMBLE TRAINING DONE ==="
