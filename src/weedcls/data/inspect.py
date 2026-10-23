"""Dataset audit: size, label format, class balance, capture sessions, near-duplicates.

Writes reports/metrics/data_summary.json, a few figures, and data/features/ helper tables
(per-image stats and near-duplicate pairs) that the split builder reuses.
"""
import argparse
import json
import re
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from weedcls.logging_utils import get_logger

log = get_logger("inspect")

_NAME = re.compile(r"^(\d{8})-(\d{6})-(\d+)\.jpg$")


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["Filename", "Label", "Species"]
    if list(df.columns) != expected:
        raise ValueError(f"unexpected label columns {list(df.columns)}, wanted {expected}")
    parsed = df["Filename"].str.extract(_NAME)
    bad = parsed[0].isna().sum()
    if bad:
        log.warning("%d filenames do not match YYYYMMDD-HHMMSS-k.jpg", bad)
    df["date"] = parsed[0]
    df["time"] = parsed[1]
    df["cam"] = parsed[2]
    df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S", errors="coerce")
    return df


def scan_images(df: pd.DataFrame, img_dir: Path) -> pd.DataFrame:
    rows = []
    for fn in tqdm(df["Filename"], desc="scanning images"):
        rec = {"Filename": fn, "ok": False}
        try:
            with Image.open(img_dir / fn) as im:
                im.load()
                rgb = im.convert("RGB")
                arr = np.asarray(rgb, dtype=np.float32) / 255.0
                rec.update(
                    ok=True,
                    w=im.width,
                    h=im.height,
                    mode=im.mode,
                    bytes=(img_dir / fn).stat().st_size,
                    brightness=float(arr.mean()),
                    r_mean=float(arr[..., 0].mean()),
                    g_mean=float(arr[..., 1].mean()),
                    b_mean=float(arr[..., 2].mean()),
                    phash=str(imagehash.phash(rgb)),
                )
        except Exception as e:  # noqa: BLE001  corrupt/missing file: record and keep going
            rec["error"] = repr(e)
            log.warning("could not read %s: %s", fn, e)
        rows.append(rec)
    return pd.DataFrame(rows)


def near_duplicate_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 400):
    """All index pairs (i<j) with Hamming distance <= max_dist between 64-bit hashes."""
    h = hashes.astype(np.uint64)
    n = len(h)
    pairs = []
    for start in range(0, n, chunk):
        block = h[start : start + chunk]
        dist = np.bitwise_count(block[:, None] ^ h[None, :]).astype(np.uint8)
        ii, jj = np.nonzero(dist <= max_dist)
        for a, b in zip(ii + start, jj, strict=True):
            if a < b:
                pairs.append((int(a), int(b), int(dist[a - start, b])))
    return pairs


def _save(fig, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--max-hash-dist", type=int, default=6)
    args = ap.parse_args()
    root, feat, rep = Path(args.root), Path(args.features), Path(args.reports)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    (rep / "metrics").mkdir(parents=True, exist_ok=True)
    feat.mkdir(parents=True, exist_ok=True)

    df = load_labels(root / "labels" / "labels.csv")
    log.info("labels: %d rows", len(df))
    stats = scan_images(df, root / "images")
    df = df.merge(stats, on="Filename")
    df.to_csv(feat / "image_stats.csv", index=False)

    ok = df[df["ok"]].reset_index(drop=True)
    names = df.groupby("Label")["Species"].first().sort_index().to_dict()
    counts = df["Label"].value_counts().sort_index()
    summary = {
        "n_labels": len(df),
        "n_readable": int(df["ok"].sum()),
        "n_unreadable": int((~df["ok"]).sum()),
        "label_columns": ["Filename", "Label", "Species"],
        "label_map": {int(k): v for k, v in names.items()},
        "class_counts": {names[k]: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": round(counts.max() / counts.min(), 2),
        "negative_fraction": round(float(counts[counts.index == 8].sum() / len(df)), 4),
        "resolutions": ok.groupby(["w", "h"]).size().rename("n").reset_index().to_dict("records"),
        "modes": ok["mode"].value_counts().to_dict(),
        "file_kb_mean": round(float(ok["bytes"].mean() / 1024), 1),
        "duplicate_filenames": int(df["Filename"].duplicated().sum()),
        "pixel_mean_rgb_all_images": [
            round(float(ok[c].mean()), 4) for c in ("r_mean", "g_mean", "b_mean")
        ],
    }

    by_date = df.groupby("date").size()
    summary["n_capture_dates"] = int(by_date.shape[0])
    summary["images_per_date"] = {k: int(v) for k, v in by_date.items()}
    summary["cam_ids"] = {str(k): int(v) for k, v in df["cam"].value_counts().items()}
    xt = pd.crosstab(df["date"], df["Species"])
    summary["classes_present_per_date"] = (xt > 0).sum(axis=1).astype(int).to_dict()
    summary["date_range"] = [str(df["datetime"].min()), str(df["datetime"].max())]

    hashes = ok["phash"].map(lambda s: int(s, 16)).to_numpy(dtype=np.uint64)
    pairs = near_duplicate_pairs(hashes, args.max_hash_dist)
    pdf = pd.DataFrame(pairs, columns=["i", "j", "dist"])
    for side, col in (("a", "i"), ("b", "j")):
        pdf[f"file_{side}"] = ok.loc[pdf[col], "Filename"].to_numpy()
        pdf[f"label_{side}"] = ok.loc[pdf[col], "Label"].to_numpy()
        pdf[f"date_{side}"] = ok.loc[pdf[col], "date"].to_numpy()
    pdf.drop(columns=["i", "j"]).to_csv(feat / "near_dup_pairs.csv", index=False)
    dup = {}
    for t in sorted({0, 2, 4, args.max_hash_dist}):
        sub = pdf[pdf["dist"] <= t]
        dup[f"hamming<={t}"] = {
            "pairs": len(sub),
            "cross_class_pairs": int((sub["label_a"] != sub["label_b"]).sum()),
            "cross_date_pairs": int((sub["date_a"] != sub["date_b"]).sum()),
        }
    summary["near_duplicates_phash64"] = dup

    (rep / "metrics" / "data_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote data_summary.json")

    order = list(counts.sort_values(ascending=False).index)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([names[k] for k in order], [counts[k] for k in order], color="#4a7c59")
    for y, k in enumerate(order):
        ax.text(counts[k] + 100, y, str(counts[k]), va="center", fontsize=8)
    ax.set_xlabel("images")
    ax.set_title("DeepWeeds class distribution")
    _save(fig, rep / "figures" / "class_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    xt.plot(kind="bar", stacked=True, ax=ax, width=0.9, colormap="tab10")
    ax.set_ylabel("images")
    ax.set_title("Images per capture date, by class")
    ax.legend(fontsize=6, ncol=3)
    ax.tick_params(axis="x", labelsize=5)
    _save(fig, rep / "figures" / "images_per_date.png")

    labs = sorted(names)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([ok.loc[ok["Label"] == k, "brightness"] for k in labs],
               tick_labels=[names[k] for k in labs])
    ax.set_ylabel("mean pixel intensity (0-1)")
    ax.set_title("Brightness per class")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    _save(fig, rep / "figures" / "brightness_by_class.png")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(labs), 8, figsize=(12, 1.6 * len(labs)))
    for r, k in enumerate(labs):
        picks = rng.choice(ok.index[ok["Label"] == k], 8, replace=False)
        for c, idx in enumerate(picks):
            axes[r, c].imshow(Image.open(root / "images" / ok.loc[idx, "Filename"]))
            axes[r, c].axis("off")
        axes[r, 0].set_title(names[k], fontsize=7, loc="left")
    _save(fig, rep / "figures" / "sample_grid.png", dpi=110)
    log.info("figures saved")


if __name__ == "__main__":
    main()
