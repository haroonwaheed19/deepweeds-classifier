"""Out-of-fold linear-probe scores on TRAIN: which labels does the model most strongly disagree with?

Folds are grouped by capture session, so a sample is never scored by a model that saw its session.
Test is not used.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weedcls.data.dataset import CLASS_NAMES

z = np.load("data/features/efficientnet_b0_224.npz", allow_pickle=True)
idx = {f: i for i, f in enumerate(z["files"])}
tr = pd.read_csv("data/splits/train.csv")
X, y, g = z["X"][[idx[f] for f in tr.Filename]], tr.Label.to_numpy(), tr.group.to_numpy()

oof = np.zeros((len(tr), 9))
for a, b in GroupKFold(n_splits=5).split(X, y, g):
    sc = StandardScaler().fit(X[a])
    clf = LogisticRegression(C=0.003, max_iter=300, tol=1e-3).fit(sc.transform(X[a]), y[a])
    p = clf.predict_proba(sc.transform(X[b]))
    oof[np.ix_(b, clf.classes_)] = p   # a class absent from a fold's train part stays at 0
tr["p_true"] = oof[np.arange(len(tr)), y]
tr["p_pred"] = oof.max(1)
tr["pred"] = oof.argmax(1)
sus = tr[(tr.pred != tr.Label) & (tr.p_pred > 0.9)].sort_values("p_true")
print("train images:", len(tr), "| confidently contradicted (p_pred>0.9):", len(sus),
      f"({len(sus) / len(tr):.1%})")
print(sus.groupby("Species").size().sort_values(ascending=False).to_dict())
top = sus.head(40).reset_index(drop=True)
top.insert(0, "id", range(1, 41))
top[["id", "Filename", "Species", "pred", "p_true", "p_pred"]].assign(verdict="").to_csv(
    "reports/label_noise_candidates_template.csv", index=False)
for s in range(2):
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    for ax, (_, r) in zip(axes.ravel(), top.iloc[s * 20:(s + 1) * 20].iterrows(), strict=True):
        ax.imshow(Image.open(f"data/raw/images/{r.Filename}"))
        ax.set_title(f"#{r.id} label {r.Species}\nmodel says {CLASS_NAMES[r.pred]} ({r.p_pred:.2f})", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"reports/figures/label_noise_sheet_{s + 1}.png", dpi=75)
    plt.close(fig)
Path("reports/metrics/label_noise_oof.json").write_text(json.dumps(
    {"n_train": len(tr), "n_confidently_contradicted": len(sus),
     "by_label": sus.groupby("Species").size().to_dict()}, indent=2))
