"""Per-channel mean/std over the TRAIN split only (val/test must not influence preprocessing)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def channel_stats(csv_path: Path, img_dir: Path) -> dict:
    n, s, s2 = 0, np.zeros(3), np.zeros(3)
    for fn in tqdm(pd.read_csv(csv_path)["Filename"], desc="channel stats"):
        a = np.asarray(Image.open(img_dir / fn).convert("RGB"), dtype=np.float64) / 255.0
        s += a.sum((0, 1))
        s2 += (a**2).sum((0, 1))
        n += a.shape[0] * a.shape[1]
    mean = s / n
    std = np.sqrt(s2 / n - mean**2)
    return {"mean": [round(float(x), 4) for x in mean], "std": [round(float(x), 4) for x in std]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/splits/train.csv")
    ap.add_argument("--img-dir", default="data/raw/images")
    ap.add_argument("--out", default="reports/metrics/norm_stats.json")
    a = ap.parse_args()
    res = channel_stats(Path(a.train_csv), Path(a.img_dir))
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(res)
