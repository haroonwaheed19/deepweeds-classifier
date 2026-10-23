#!/usr/bin/env bash
# Cheap HP sweep for the fine-tuned MobileNetV3-L: 30% of train, 3 epochs (1 head-only + 2 with
# the last N blocks unfrozen). Pick by val macro-F1 (scripts/summarize_runs.py).
set -e
for lr in 3.0e-4 1.0e-3; do
  for n in 2 4; do
    python -m weedcls.train.train --config configs/finetune.yaml \
      --override train.epochs=3 train.subset_fraction=0.3 train.lr=$lr model.unfreeze_last=$n \
      --run-name "sweep_ft_lr${lr}_last${n}" --runs-dir runs/sweep_finetune
  done
done

# Added after the first 4 runs: the best point sat at the edge of the grid (lr 1e-3, last 4),
# so check one deeper unfreeze before committing to the long run.
# python -m weedcls.train.train --config configs/finetune.yaml \
#   --override train.epochs=3 train.subset_fraction=0.3 train.lr=1.0e-3 model.unfreeze_last=8 \
#   --run-name sweep_ft_lr1.0e-3_last8 --runs-dir runs/sweep_finetune
