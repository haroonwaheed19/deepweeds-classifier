"""Save a figure of original vs augmented samples so the augmentation choices can be eyeballed."""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from weedcls.config import load_config
from weedcls.data.transforms import build_train_transform
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
ap.add_argument("--out", default="reports/figures/augmentation_examples.png")
args = ap.parse_args()
cfg = load_config(args.config)
random.seed(1)
torch.manual_seed(1)

df = pd.read_csv(Path(cfg["paths"]["splits"]) / "train.csv")
rows = df.groupby("Label").sample(1, random_state=3).head(6)
tf = build_train_transform(224, IMAGENET_MEAN, IMAGENET_STD, cfg["augment"])
mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
std = torch.tensor(IMAGENET_STD)[:, None, None]

fig, axes = plt.subplots(len(rows), 5, figsize=(10, 2 * len(rows)))
for r, (_, row) in enumerate(rows.iterrows()):
    img = Image.open(Path(cfg["paths"]["images"]) / row["Filename"]).convert("RGB")
    axes[r, 0].imshow(img)
    axes[r, 0].set_title(row["Species"], fontsize=7, loc="left")
    for c in range(1, 5):
        axes[r, c].imshow((tf(img) * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
    for a in axes[r]:
        a.axis("off")
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110)
