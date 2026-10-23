"""Publish to Hugging Face.

    python scripts/publish_hf.py model              # upload ONNX + card to a model repo, pin the manifest
    python scripts/publish_hf.py space              # stage the serving image context and push it to a Space
    python scripts/publish_hf.py space --stage-only # just build the staging dir (CI uses this)

Needs `hf auth login` (or HF_TOKEN) for the first two forms.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "service" / "model_manifest.json"
SPACE_README = """---
title: DeepWeeds Classifier
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
---

# DeepWeeds classifier

MobileNetV3-Large fine-tuned on the DeepWeeds dataset (8 weed species + Negative), served as a
FastAPI + ONNX Runtime service on CPU.

- Upload page: `/`
- API docs: `/docs`
- `POST /predict` with a multipart `file` field returns the label, calibrated confidence and top-3.

Code, training and evaluation: see the GitHub repository linked from the project README.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "src" / "weedcls" / "inference").mkdir(parents=True)
    (out / "scripts").mkdir()
    shutil.copy(ROOT / "service" / "Dockerfile", out / "Dockerfile")
    shutil.copytree(ROOT / "service", out / "service",
                    ignore=shutil.ignore_patterns("__pycache__", "Dockerfile", "*.pyc"))
    for rel in ("__init__.py", "preprocess.py", "inference/__init__.py", "inference/predictor.py"):
        shutil.copy(ROOT / "src" / "weedcls" / rel, out / "src" / "weedcls" / rel)
    shutil.copy(ROOT / "scripts" / "fetch_model.py", out / "scripts" / "fetch_model.py")
    (out / "README.md").write_text(SPACE_README)
    return out


def publish_model(repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    art = ROOT / "artifacts"
    card = (
        "---\nlicense: cc-by-4.0\ntags: [image-classification, onnx, agriculture]\n---\n\n"
        "# DeepWeeds MobileNetV3-Large (ONNX)\n\nFine-tuned on DeepWeeds (Olsen et al., 2019). "
        "Preprocessing, calibration constants and class list are in `model_card.json`.\n"
    )
    (art / "README.md").write_text(card)
    info = api.upload_folder(folder_path=str(art), repo_id=repo, repo_type="model",
                             allow_patterns=["model.onnx", "model_card.json", "README.md"],
                             commit_message="Add DeepWeeds MobileNetV3-L ONNX model v1")
    revision = info.oid
    MANIFEST.write_text(json.dumps({
        "repo": repo, "revision": revision,
        "files": {"model.onnx": sha256(art / "model.onnx"), "model_card.json": sha256(art / "model_card.json")},
    }, indent=2) + "\n")
    print(f"model published: https://huggingface.co/{repo}/tree/{revision}\nmanifest updated: {MANIFEST}")


def publish_space(repo: str, out: Path) -> None:
    from huggingface_hub import HfApi

    if "PENDING" in MANIFEST.read_text():
        raise SystemExit("run `publish_hf.py model` first so the manifest pins a model revision")
    staged = stage(out)
    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(staged), repo_id=repo, repo_type="space",
                      commit_message="Deploy DeepWeeds service")
    print(f"space pushed: https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["model", "space"])
    ap.add_argument("--model-repo", help="e.g. username/deepweeds-mobilenetv3-onnx")
    ap.add_argument("--space-repo", help="e.g. username/deepweeds-classifier")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--out", default="build/space")
    a = ap.parse_args()
    if a.target == "model":
        if not a.model_repo:
            raise SystemExit("--model-repo is required")
        publish_model(a.model_repo)
    elif a.stage_only:
        print(f"staged at {stage(Path(a.out))}")
    else:
        if not a.space_repo:
            raise SystemExit("--space-repo is required")
        publish_space(a.space_repo, Path(a.out))
