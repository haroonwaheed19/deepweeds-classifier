"""Grad-CAM of the final model on selected test errors (ids from reports/error_tags.csv).

CAM for the *predicted* (wrong) class on the last conv feature map of MobileNetV3-L.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image

from weedcls.data.transforms import build_eval_transform
from weedcls.eval.evaluate import load_torch

IDS = [1, 9, 12, 5, 13, 18, 22, 35, 40, 36, 47, 10]
tags = pd.read_csv("reports/error_tags.csv").set_index("id")
model, size, mean, std = load_torch(Path("runs/m3_mobilenet_ft"))
tf = build_eval_transform(size, mean, std)

acts, grads = {}, {}
layer = model.features[-1]
layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

fig, axes = plt.subplots(3, 8, figsize=(20, 8.5))
for k, i in enumerate(IDS):
    r = tags.loc[i]
    img = Image.open(f"data/raw/images/{r.Filename}").convert("RGB")
    x = tf(img)[None]
    model.zero_grad()
    logits = model(x)
    target = int(logits.argmax(1))
    logits[0, target].backward()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = np.asarray(img.resize((size, size)))
    row, col = divmod(k, 4)
    axes[row, col * 2].imshow(base)
    axes[row, col * 2].set_title(f"#{i} true {r.Species}\npred {r.pred_name} [{r.tag}]", fontsize=8)
    axes[row, col * 2 + 1].imshow(base)
    axes[row, col * 2 + 1].imshow(cam, cmap="jet", alpha=0.45)
    axes[row, col * 2 + 1].set_title("Grad-CAM (predicted class)", fontsize=8)
for a in axes.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig("reports/figures/gradcam_errors.png", dpi=80)
