"""Download the ONNX model + model card from the Hugging Face model repo named in
service/model_manifest.json, pinned to a commit and verified by sha256.

Stdlib only, so it can run inside the slim serving image at build time:

    python scripts/fetch_model.py --out artifacts
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "service" / "model_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(manifest_path: Path, out: Path) -> None:
    m = json.loads(manifest_path.read_text())
    if "PENDING" in json.dumps(m):
        sys.exit(f"{manifest_path} has not been filled in yet; run scripts/publish_hf.py model first")
    out.mkdir(parents=True, exist_ok=True)
    for name, want in m["files"].items():
        dest = out / name
        if dest.exists() and sha256(dest) == want:
            print(f"{name}: already present and verified")
            continue
        url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{name}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        got = sha256(dest)
        if got != want:
            dest.unlink()
            sys.exit(f"checksum mismatch for {name}: expected {want}, got {got}")
        print(f"{name}: ok ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    fetch(Path(a.manifest), Path(a.out))
