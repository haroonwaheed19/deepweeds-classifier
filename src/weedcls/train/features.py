"""Run a frozen pretrained backbone once over every image and cache the embeddings.

The linear-probe sweep then costs seconds per config instead of a forward pass per config.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from weedcls.data.dataset import WeedDataset
from weedcls.data.transforms import build_eval_transform
from weedcls.logging_utils import get_logger
from weedcls.models.backbones import build_backbone, strip_head
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD

log = get_logger("features")


@torch.no_grad()
def extract(name: str, size: int, labels_csv: Path, img_dir: Path, batch: int = 64, workers: int = 2):
    model = strip_head(build_backbone(name, pretrained=True), name).eval()
    ds = WeedDataset(labels_csv, img_dir, build_eval_transform(size, IMAGENET_MEAN, IMAGENET_STD))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers)
    feats, t0 = [], time.time()
    for i, (x, _) in enumerate(loader):
        feats.append(model(x).numpy().astype(np.float32))
        if i % 20 == 0:
            log.info("%s@%d: %d/%d images, %.0fs", name, size, min((i + 1) * batch, len(ds)),
                     len(ds), time.time() - t0)
    return np.concatenate(feats), ds.df["Filename"].to_numpy(), ds.labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--labels", default="data/raw/labels/labels.csv")
    ap.add_argument("--img-dir", default="data/raw/images")
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir) / f"{args.backbone}_{args.size}.npz"
    if out.exists():
        log.info("%s already cached, skipping", out)
        return
    X, files, y = extract(args.backbone, args.size, Path(args.labels), Path(args.img_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, files=files, y=y)
    log.info("saved %s %s", out, X.shape)
    pd.Series({"backbone": args.backbone, "size": args.size, "dim": X.shape[1]}).to_json(
        out.with_suffix(".json"))


if __name__ == "__main__":
    main()
