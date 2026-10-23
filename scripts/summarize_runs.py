"""Collect best val metrics + config of every run under a directory into one table."""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir")
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = []
for d in sorted(Path(args.runs_dir).iterdir()):
    if not (d / "summary.json").exists():
        continue
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    m = pd.read_csv(d / "metrics.csv")
    best = m.loc[m["val_macro_f1"].idxmax()]
    rows.append({
        "run": d.name, "lr": cfg["train"]["lr"], "imbalance": cfg["imbalance"],
        "unfreeze_last": cfg["model"].get("unfreeze_last"), "epochs_run": len(m),
        "best_epoch": int(best["epoch"]), "val_macro_f1": best["val_macro_f1"],
        "val_acc": best["val_acc"], "val_weed_recall": best["val_weed_recall"],
        "val_false_alarm": best["val_false_alarm"], "min_per_epoch": round(m["seconds"].mean() / 60, 1),
    })
df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
print(df.round(4).to_string(index=False))
if args.out:
    df.to_csv(args.out, index=False)
