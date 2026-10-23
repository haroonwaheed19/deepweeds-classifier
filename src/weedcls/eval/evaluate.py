"""Test-set evaluation of the final models. Run once; test labels never feed back into training
or into any threshold (T, bias and the uncertainty cut-off were all fit on validation).

    python -m weedcls.eval.evaluate
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from weedcls.data.dataset import CLASS_NAMES, WeedDataset, make_loader
from weedcls.data.transforms import build_eval_transform
from weedcls.eval.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_diff
from weedcls.logging_utils import get_logger
from weedcls.models.registry import build_model
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD
from weedcls.train.engine import predict_logits

log = get_logger("evaluate")


def load_torch(run_dir: Path):
    ck = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model({**cfg["model"], "pretrained": False}, len(CLASS_NAMES))
    model.load_state_dict(ck["model"])
    if cfg["norm"] == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        s = json.loads(Path("reports/metrics/norm_stats.json").read_text())
        mean, std = s["mean"], s["std"]
    return model.eval(), cfg["model"]["img_size"], mean, std


def torch_logits(run_dir: Path, csv: Path, img_dir: str):
    model, size, mean, std = load_torch(run_dir)
    ds = WeedDataset(csv, img_dir, build_eval_transform(size, mean, std))
    return predict_logits(model, make_loader(ds, 64, False, 2))[0], ds.labels, model


def latency_ms(fn, n: int = 20) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--img-dir", default="data/raw/images")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="reports/metrics")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    test = pd.read_csv(Path(args.splits) / "test.csv")
    y = test["Label"].to_numpy()
    groups = test["group"].to_numpy()
    cal = json.loads((out / "calibration.json").read_text())

    preds, _extras = {}, {}
    preds["M0 majority"] = np.full(len(y), 8)

    z1, y1, m1 = torch_logits(Path("runs/m1_small_cnn"), Path(args.splits) / "test.csv", args.img_dir)
    assert (y1 == y).all()
    preds["M1 small CNN (scratch)"] = z1.argmax(1)

    probe = joblib.load("runs/linear_probe/best.joblib")
    zz = np.load(Path(args.features) / f"{probe['backbone']}_{probe['size']}.npz", allow_pickle=True)
    idx = {f: i for i, f in enumerate(zz["files"])}
    rows = np.array([idx[f] for f in test["Filename"]])
    Xt = probe["scaler"].transform(zz["X"][rows])
    preds["M2 probe (EfficientNet-B0 frozen)"] = probe["clf"].predict(Xt)

    z3, y3, m3 = torch_logits(Path("runs/m3_mobilenet_ft"), Path(args.splits) / "test.csv", args.img_dir)
    assert (y3 == y).all()
    preds["M3 fine-tuned (raw argmax)"] = z3.argmax(1)
    zc = z3 / cal["temperature"]
    zc[:, 8] += cal["negative_logit_bias"]
    preds["M3 fine-tuned + calibration"] = zc.argmax(1)

    results = {}
    for name, p in preds.items():
        m = compute_metrics(y, p)
        lo, hi = bootstrap_ci(y, p, groups, "macro_f1", args.n_boot)
        m["macro_f1_ci95"] = [lo, hi]
        results[name] = m
        log.info("%s: macro-F1 %.3f [%.3f, %.3f] acc %.3f weed recall %.3f false alarm %.3f", name,
                 m["macro_f1"], lo, hi, m["accuracy"], m["weed_recall"], m["false_alarm_rate"])

    final, probe_name = "M3 fine-tuned + calibration", "M2 probe (EfficientNet-B0 frozen)"
    for a, b in ((final, probe_name), (final, "M1 small CNN (scratch)"),
                 (final, "M3 fine-tuned (raw argmax)")):
        d, lo, hi = paired_bootstrap_diff(y, preds[a], preds[b], groups, "macro_f1", args.n_boot)
        results.setdefault("paired_diffs", {})[f"{a}  minus  {b}"] = {"mean": d, "ci95": [lo, hi]}
        log.info("macro-F1 diff %s - %s: %.3f [%.3f, %.3f]", a, b, d, lo, hi)

    # cost side of the table
    x = torch.randn(1, 3, 224, 224)
    x128 = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        cost = {
            "M1 small CNN (scratch)": {"params_m": sum(p.numel() for p in m1.parameters()) / 1e6,
                                       "torch_ms_per_image": latency_ms(lambda: m1(x128))},
            "M3 fine-tuned + calibration": {
                "params_m": sum(p.numel() for p in m3.parameters()) / 1e6,
                "torch_ms_per_image": latency_ms(lambda: m3(x)),
                "onnx_mb": round(Path("artifacts/model.onnx").stat().st_size / 1e6, 1)},
        }
    results["cost"] = cost

    # per-image predictions of the final model, for error analysis
    probs = torch.softmax(torch.from_numpy(zc), 1).numpy()
    det = test[["Filename", "Label", "Species", "group"]].copy()
    det["pred"] = preds[final]
    det["pred_name"] = [CLASS_NAMES[i] for i in det["pred"]]
    det["confidence"] = probs.max(1)
    det["p_weed"] = 1 - probs[:, 8]
    det["correct"] = det["pred"] == det["Label"]
    det["date"] = det["Filename"].str[:8]
    det["cam"] = det["Filename"].str.extract(r"-(\d+)\.jpg")[0]
    for k, c in enumerate(CLASS_NAMES):
        det[f"p_{c}"] = probs[:, k]
    det.to_csv(out / "test_predictions_m3.csv", index=False)
    (out / "test_results.json").write_text(json.dumps(results, indent=2))
    log.info("saved test_results.json and test_predictions_m3.csv")


if __name__ == "__main__":
    main()
