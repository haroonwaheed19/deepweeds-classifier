"""Export a trained checkpoint to ONNX + model_card.json and verify PyTorch/ONNX parity.

    python -m weedcls.inference.export_onnx --run runs/m3_mobilenet --out artifacts
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from weedcls.data.dataset import CLASS_NAMES
from weedcls.logging_utils import get_logger
from weedcls.models.registry import build_model
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD

log = get_logger("export")


def export(model: torch.nn.Module, size: int, onnx_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )


def parity(model: torch.nn.Module, onnx_path: Path, size: int, n: int = 8, atol: float = 1e-3) -> float:
    x = torch.randn(n, 3, size, size)
    with torch.no_grad():
        ref = model.eval()(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing best.pt")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--negative-bias", type=float, default=0.0)
    ap.add_argument("--uncertain-threshold", type=float, default=0.0)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model({**cfg["model"], "pretrained": False}, len(CLASS_NAMES))
    model.load_state_dict(ck["model"])
    size = cfg["model"]["img_size"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export(model, size, out / "model.onnx")
    diff = parity(model, out / "model.onnx", size)
    log.info("parity max |diff| = %.2e", diff)

    if cfg["norm"] == "imagenet":
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    else:
        stats = json.loads(Path("reports/metrics/norm_stats.json").read_text())
        mean, std = stats["mean"], stats["std"]
    card = {
        "model_version": args.version, "architecture": cfg["model"]["name"], "input_size": size,
        "mean": mean, "std": std, "classes": CLASS_NAMES, "negative_index": 8,
        "temperature": args.temperature, "negative_logit_bias": args.negative_bias,
        "uncertain_threshold": args.uncertain_threshold,
        "val_macro_f1": ck.get("val_macro_f1"), "onnx_parity_max_abs_diff": diff,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    log.info("wrote %s and model_card.json", out / "model.onnx")


if __name__ == "__main__":
    main()
