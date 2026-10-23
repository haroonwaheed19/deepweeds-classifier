"""Fetch DeepWeeds (Olsen et al., Sci Rep 2019, CC BY 4.0) into data/raw/. Safe to re-run."""
import argparse
import zipfile
from pathlib import Path

import requests

from weedcls.logging_utils import get_logger

log = get_logger("download")

RAW = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/"
IMAGES_DRIVE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
EXPECTED_IMAGES = 17509
OFFICIAL_SPLIT_FILES = ["labels.csv"] + [
    f"{kind}_subset0.csv" for kind in ("train", "val", "test")
]


def fetch_labels(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_SPLIT_FILES:
        dest = out / name
        if dest.exists():
            continue
        r = requests.get(RAW + name, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        log.info("saved %s (%d bytes)", dest, len(r.content))


def fetch_images(root: Path) -> None:
    img_dir = root / "images"
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) == EXPECTED_IMAGES:
        log.info("images already present (%d), skipping", EXPECTED_IMAGES)
        return
    import gdown  # only needed here, keeps the rest of the package light

    zip_path = root / "images.zip"
    if not zip_path.exists():
        log.info("downloading images.zip (~468 MB) from Google Drive")
        got = gdown.download(id=IMAGES_DRIVE_ID, output=str(zip_path), quiet=False)
        if got is None:
            raise RuntimeError(
                "gdown failed (Drive quota?). Download images.zip manually from "
                "https://drive.google.com/file/d/" + IMAGES_DRIVE_ID + " into " + str(root)
            )
    img_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if m.lower().endswith(".jpg"):
                (img_dir / Path(m).name).write_bytes(zf.read(m))
    n = len(list(img_dir.glob("*.jpg")))
    log.info("extracted %d images", n)
    if n != EXPECTED_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_IMAGES} images, found {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    fetch_labels(root / "labels")
    fetch_images(root)


if __name__ == "__main__":
    main()
