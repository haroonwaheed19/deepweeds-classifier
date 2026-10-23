"""Does recall depend on whether the test image's capture date also appears in train for its class?"""
import json
from pathlib import Path

import pandas as pd

from weedcls.eval.metrics import bootstrap_ci

det = pd.read_csv("reports/metrics/test_predictions_m3.csv", dtype={"date": str, "cam": str})
tr = pd.read_csv("data/splits/train.csv")
tr["date"] = tr["Filename"].str[:8]
seen = {(r.Species, r.date) for r in tr.itertuples()}
w = det[det.Label != 8].copy()
w["date_seen_in_train"] = [(s, d) in seen for s, d in zip(w.Species, w.date, strict=True)]
w["found_as_weed"] = w.pred != 8

rows = []
for flag, g in w.groupby("date_seen_in_train"):
    rows.append({"date_seen_in_train_for_class": bool(flag), "n": len(g),
                 "species_recall": round(float(g.correct.mean()), 3),
                 "weed_recall": round(float(g.found_as_weed.mean()), 3)})
by_class = (w.groupby(["Species", "date_seen_in_train"]).agg(n=("correct", "size"), recall=("correct", "mean"))
            .round(3).reset_index().to_dict("records"))
out = {"overall": rows, "by_class": by_class}
for flag in (True, False):
    g = w[w.date_seen_in_train == flag]
    lo, hi = bootstrap_ci(g.Label.to_numpy(), g.pred.to_numpy(), g.group.to_numpy(), "accuracy")
    out[f"accuracy_ci95_seen={flag}"] = [round(lo, 3), round(hi, 3)]
Path("reports/metrics/date_shift_analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
