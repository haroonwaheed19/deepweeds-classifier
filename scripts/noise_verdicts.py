"""Manual verdicts on the 40 most contradicted train labels (eyeballed from label_noise_sheet_*.png)."""
import json
from pathlib import Path

import pandas as pd

# N = no target visible for a weed label (likely mislabel), P = label plausible (model wrong), A = ambiguous
V = ["P", "A", "P", "A", "N", "N", "A", "A", "N", "N", "P", "A", "A", "A", "P", "N", "A", "P", "A", "A", "A", "A", "P", "P", "A", "N", "A", "A", "P", "P", "P", "A", "A", "P", "N", "A", "P", "A", "P", "A"]
t = pd.read_csv("reports/label_noise_candidates_template.csv")
assert len(V) == len(t) == 40
t["verdict"] = V
t.to_csv("reports/label_noise_verdicts.csv", index=False)
OUT = Path("reports/metrics/label_noise_oof.json")
base = json.loads(OUT.read_text())
c = t.verdict.value_counts()
base["top40_verdicts"] = {"likely_mislabel_no_target_visible": int(c["N"]), "label_plausible_model_wrong": int(c["P"]),
                          "ambiguous": int(c["A"])}
base["note"] = "top 40 by lowest out-of-fold p(true label); not a random sample, so mislabel rate is an upper-ish bound"
OUT.write_text(json.dumps(base, indent=2))
print(base["top40_verdicts"])
print(t[t.verdict == "N"].Species.value_counts().to_dict())
