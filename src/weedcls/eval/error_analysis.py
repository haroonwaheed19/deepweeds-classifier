"""Quantitative + visual error analysis of the final model on the test set.

Reads reports/metrics/test_predictions_m3.csv and test_results.json (written by evaluate.py)
and writes figures to reports/figures/ plus reports/metrics/error_breakdown.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from weedcls.data.dataset import CLASS_NAMES

FINAL = "M3 fine-tuned + calibration"


def confusion_figure(cm: np.ndarray, path: Path) -> None:
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(9), CLASS_NAMES, fontsize=8)
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion matrix (row-normalised), final model")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_f1_figure(results: dict, path: Path) -> None:
    names = ["M1 small CNN (scratch)", "M2 probe (EfficientNet-B0 frozen)", FINAL]
    x = np.arange(9)
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, n in enumerate(names):
        f1 = [results[n]["per_class"][str(c)]["f1"] for c in range(9)]
        ax.bar(x + (k - 1) * 0.27, f1, 0.27, label=n.split(" (")[0])
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test F1")
    ax.legend(fontsize=8)
    ax.set_title("Per-class F1 on test")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_grid(df: pd.DataFrame, img_dir: Path, title: str, path: Path, n: int = 8) -> None:
    if df.empty:
        return
    df = df.head(n)
    cols = 4
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), df.iterrows(), strict=False):
        ax.imshow(Image.open(img_dir / r["Filename"]))
        ax.set_title(f"true {r['Species']}\npred {r['pred_name']} ({r['confidence']:.2f})\n{r['Filename'][:15]}",
                     fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/metrics")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--img-dir", default="data/raw/images")
    args = ap.parse_args()
    mdir, fdir, img_dir = Path(args.metrics), Path(args.figures), Path(args.img_dir)
    det = pd.read_csv(mdir / "test_predictions_m3.csv", dtype={"date": str, "cam": str})
    results = json.loads((mdir / "test_results.json").read_text())

    confusion_figure(np.array(results[FINAL]["confusion_matrix"]), fdir / "confusion_test.png")
    per_class_f1_figure(results, fdir / "per_class_f1_test.png")

    errs = det[~det["correct"]].sort_values("confidence", ascending=False)
    pairs = errs.groupby(["Species", "pred_name"]).size().sort_values(ascending=False)
    breakdown = {
        "n_test": len(det), "n_errors": len(errs),
        "top_confusions": [{"true": a, "pred": b, "n": int(n)} for (a, b), n in pairs.head(12).items()],
    }

    # does error depend on session or on the filename suffix?
    by_date = det.groupby("date").agg(n=("correct", "size"), acc=("correct", "mean"),
                                      classes=("Species", "nunique")).query("n >= 30")
    breakdown["accuracy_by_date_n>=30"] = by_date.sort_values("acc").round(3).reset_index().to_dict("records")
    breakdown["accuracy_by_suffix"] = (
        det.groupby("cam").agg(n=("correct", "size"), acc=("correct", "mean")).round(3).reset_index()
        .to_dict("records"))
    breakdown["per_class"] = {
        c: {"n": int((det.Species == c).sum()), "recall": round(float(det.loc[det.Species == c, "correct"].mean()), 3)}
        for c in CLASS_NAMES}

    # weeds called Negative (misses) and Negatives called weed (false alarms)
    missed = det[(det.Label != 8) & (det.pred == 8)]
    false_alarm = det[(det.Label == 8) & (det.pred != 8)]
    breakdown["missed_weeds"] = {"n": len(missed), "by_class": missed.Species.value_counts().to_dict()}
    breakdown["false_alarms"] = {"n": len(false_alarm), "by_pred": false_alarm.pred_name.value_counts().to_dict()}
    wrong_species = det[(det.Label != 8) & (det.pred != 8) & (~det.correct)]
    breakdown["weed_found_wrong_species"] = {
        "n": len(wrong_species), "share_of_weed_images_found": round(
            len(wrong_species) / max(int(((det.Label != 8) & (det.pred != 8)).sum()), 1), 3)}

    err_conf = errs["confidence"]
    breakdown["error_confidence_quantiles"] = {q: round(float(err_conf.quantile(q)), 3) for q in (0.25, 0.5, 0.75)}
    breakdown["uncertain_flag_catches_errors"] = {
        "share_of_errors_below_0.619": round(float((err_conf < 0.619).mean()), 3),
        "share_of_correct_below_0.619": round(float((det.loc[det.correct, "confidence"] < 0.619).mean()), 3)}
    (mdir / "error_breakdown.json").write_text(json.dumps(breakdown, indent=2))

    # image grids
    error_grid(missed.sort_values("confidence", ascending=False), img_dir,
               "Weeds predicted as Negative (most confident first)", fdir / "errors_missed_weeds.png", 12)
    error_grid(false_alarm.sort_values("confidence", ascending=False), img_dir,
               "Negatives predicted as a weed (most confident first)", fdir / "errors_false_alarms.png", 8)
    for (a, b) in pairs.head(3):
        sub = errs[(errs.Species == a) & (errs.pred_name == b)]
        error_grid(sub, img_dir, f"{a} predicted as {b}",
                   fdir / f"errors_{a.replace(' ', '_')}_as_{b.replace(' ', '_')}.png", 8)
    print(json.dumps({k: breakdown[k] for k in ("n_errors", "top_confusions", "missed_weeds", "false_alarms",
                                                "weed_found_wrong_species", "error_confidence_quantiles",
                                                "uncertain_flag_catches_errors", "per_class")}, indent=1))


if __name__ == "__main__":
    main()
