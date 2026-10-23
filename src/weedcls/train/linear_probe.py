"""Logistic regression on cached frozen-backbone embeddings (the cheap transfer-learning test).

Sweeps backbone x C x class-weighting on the grouped split and picks by val macro-F1, then
re-scores the winning config on the paper's random split to put a number on the leakage gap.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from weedcls.eval.metrics import compute_metrics
from weedcls.logging_utils import get_logger

log = get_logger("probe")


def load_features(feat_dir: Path, backbone: str, size: int):
    z = np.load(feat_dir / f"{backbone}_{size}.npz", allow_pickle=True)
    return z["X"], list(z["files"]), z["y"]


def rows_for(files_index: dict, csv_path: Path) -> np.ndarray:
    return np.array([files_index[f] for f in pd.read_csv(csv_path)["Filename"]])


def sample_weights(y: np.ndarray, mode: str):
    if mode == "none":
        return None
    freq = np.bincount(y, minlength=9).astype(float)
    w = 1.0 / freq ** (0.5 if mode == "sqrt" else 1.0)
    w *= len(y) / (w * freq).sum()
    return w[y]


def fit_eval(X, y, tr, va, C, cw):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, max_iter=300, tol=1e-3)
    clf.fit(sc.transform(X[tr]), y[tr], sample_weight=sample_weights(y[tr], cw))
    pred = clf.predict(sc.transform(X[va]))
    return sc, clf, compute_metrics(y[va], pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--backbones", nargs="+", default=["mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.0003, 0.003, 0.03, 0.3])
    ap.add_argument("--out", default="reports/metrics")
    args = ap.parse_args()
    feat, spl, out = Path(args.features), Path(args.splits), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, best = [], None
    for bb in args.backbones:
        X, files, y = load_features(feat, bb, args.size)
        idx = {f: i for i, f in enumerate(files)}
        tr, va = rows_for(idx, spl / "train.csv"), rows_for(idx, spl / "val.csv")
        for cw in ("none", "sqrt", "balanced"):
            for C in args.Cs:
                t0 = time.time()
                sc, clf, m = fit_eval(X, y, tr, va, C, cw)
                row = {"backbone": bb, "C": C, "class_weight": cw, "val_macro_f1": m["macro_f1"],
                       "val_acc": m["accuracy"], "val_bal_acc": m["balanced_accuracy"],
                       "val_weed_recall": m["weed_recall"], "val_false_alarm": m["false_alarm_rate"],
                       "seconds": round(time.time() - t0, 1)}
                results.append(row)
                log.info("%s", row)
                if best is None or m["macro_f1"] > best["val_macro_f1"]:
                    best = {**row, "_sc": sc, "_clf": clf}
    pd.DataFrame(results).to_csv(out / "linear_probe_sweep.csv", index=False)

    # leakage check: same backbone/C/weights, but the paper's random 60/20/20 split
    X, files, y = load_features(feat, best["backbone"], args.size)
    idx = {f: i for i, f in enumerate(files)}
    otr, ova = rows_for(idx, spl / "official_train.csv"), rows_for(idx, spl / "official_val.csv")
    _, _, m_off = fit_eval(X, y, otr, ova, best["C"], best["class_weight"])
    summary = {
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "grouped_val_macro_f1": best["val_macro_f1"],
        "official_random_val_macro_f1": m_off["macro_f1"],
        "leakage_gap_macro_f1": m_off["macro_f1"] - best["val_macro_f1"],
        "grouped_val_acc": best["val_acc"],
        "official_random_val_acc": m_off["accuracy"],
    }
    (out / "linear_probe_best.json").write_text(json.dumps(summary, indent=2))
    Path("runs/linear_probe").mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": best["_sc"], "clf": best["_clf"], "backbone": best["backbone"],
                 "size": args.size}, "runs/linear_probe/best.joblib")
    log.info("best: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
