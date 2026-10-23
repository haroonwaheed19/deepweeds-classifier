"""Random sample of test errors on numbered contact sheets, for manual failure-mode tagging."""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

det = pd.read_csv("reports/metrics/test_predictions_m3.csv")
errs = det[~det.correct].sample(60, random_state=7).reset_index(drop=True)
errs.insert(0, "id", range(1, 61))
errs[["id", "Filename", "Species", "pred_name", "confidence"]].assign(tag="").to_csv(
    "reports/error_tags_template.csv", index=False)
for s in range(3):
    part = errs.iloc[s * 20:(s + 1) * 20]
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    for ax, (_, r) in zip(axes.ravel(), part.iterrows(), strict=True):
        ax.imshow(Image.open(f"data/raw/images/{r.Filename}"))
        ax.set_title(f"#{r.id}  true {r.Species}\npred {r.pred_name} ({r.confidence:.2f})", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"reports/figures/error_sheet_{s + 1}.png", dpi=75)
    plt.close(fig)
print(errs.groupby(["Species", "pred_name"]).size().sort_values(ascending=False).head(8))
