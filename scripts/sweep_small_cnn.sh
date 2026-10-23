#!/usr/bin/env bash
# Cheap sweep for the from-scratch CNN: 30% of train, 4 epochs each. Compares learning rate
# and how class imbalance is handled; winners are picked by val macro-F1 (see summarize_runs.py).
set -e
for lr in 1.0e-3 3.0e-3; do
  for imb in weights sampler none; do
    .venv/Scripts/python.exe -m weedcls.train.train --config configs/small_cnn.yaml \
      --override train.epochs=4 train.subset_fraction=0.3 train.lr=$lr imbalance=$imb \
      --run-name "sweep_cnn_lr${lr}_${imb}" --runs-dir runs/sweep_small_cnn
  done
done
