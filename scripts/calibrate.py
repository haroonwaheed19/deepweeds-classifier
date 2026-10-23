"""Fit temperature, Negative-logit bias and the 'uncertain' threshold on VALIDATION logits only."""
import argparse
import json
from pathlib import Path

import numpy as np

from weedcls.eval.calibration import (
    choose_uncertain_threshold,
    expected_calibration_error,
    fit_negative_bias,
    fit_temperature,
    reliability_plot,
)
from weedcls.eval.metrics import compute_metrics
from weedcls.train.engine import softmax_np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--target-acc", type=float, default=0.90)
ap.add_argument("--out", default="reports/metrics/calibration.json")
args = ap.parse_args()

z = np.load(Path(args.run) / "val_logits.npz")
logits, y = z["logits"], z["y"]
T = fit_temperature(logits, y)
bias, curve = fit_negative_bias(logits, y, temperature=T)

p_raw, p_T = softmax_np(logits), softmax_np(logits, T)
zb = logits / T
zb[:, 8] += bias
p_final = softmax_np(zb * 1.0)
thr, coverage, acc_above = choose_uncertain_threshold(p_final, y, args.target_acc)

before = compute_metrics(y, logits.argmax(1))
after = compute_metrics(y, zb.argmax(1))
res = {
    "temperature": T, "negative_logit_bias": bias,
    "ece_raw": expected_calibration_error(p_raw, y), "ece_after_T": expected_calibration_error(p_T, y),
    "ece_final": expected_calibration_error(p_final, y),
    "uncertain_threshold": thr, "coverage_above_threshold": coverage, "acc_above_threshold": acc_above,
    "val_before_bias": {k: before[k] for k in ("macro_f1", "accuracy", "weed_recall", "false_alarm_rate")},
    "val_after_bias": {k: after[k] for k in ("macro_f1", "accuracy", "weed_recall", "false_alarm_rate")},
    "bias_curve": curve,
}
Path(args.out).write_text(json.dumps(res, indent=2))
reliability_plot(p_raw, p_T, y, "reports/figures/reliability_val.png")
print(json.dumps({k: v for k, v in res.items() if k != "bias_curve"}, indent=2))
