"""Apply the manual failure-mode tags (assigned by eye from reports/figures/error_sheet_*.png)
to the sampled errors and summarise them."""
import json
from pathlib import Path

import pandas as pd

CATEGORIES = {
    "A": "small or sparse weed, missed (weed present but tiny / low contrast in litter)",
    "B": "weed species confused with another weed species",
    "C": "non-target look-alike plant (true Negative) called a weed",
    "D": "harsh shadow, overexposure or colour cast",
    "E": "no clear weed visible (possible label noise)",
    "F": "weed clearly visible but called Negative",
}
# tags for ids 1..60, in order
TAGS = ["A", "B", "F", "C", "B", "E", "C", "C", "A", "D", "F", "A", "B", "B", "A", "D", "A", "B", "B", "F", "B", "C", "A", "E", "C", "B", "A", "A", "B", "B", "D", "C", "B", "A", "C", "F", "C", "B", "A", "C", "A", "F", "F", "D", "A", "E", "F", "C", "A", "F", "F", "A", "B", "F", "D", "C", "D", "E", "A", "C"]

t = pd.read_csv("reports/error_tags_template.csv")
assert len(TAGS) == len(t) == 60
t["tag"] = TAGS
t["tag_meaning"] = t["tag"].map(CATEGORIES)
t.to_csv("reports/error_tags.csv", index=False)
counts = t["tag"].value_counts().sort_index()
summary = {k: {"n": int(counts.get(k, 0)), "share": round(float(counts.get(k, 0) / 60), 3), "meaning": v}
           for k, v in CATEGORIES.items()}
summary["_by_true_class"] = pd.crosstab(t["Species"], t["tag"]).to_dict(orient="index")
Path("reports/metrics/error_tags_summary.json").write_text(json.dumps(summary, indent=2))
for k, v in summary.items():
    if not k.startswith("_"):
        print(k, v["n"], f"{v['share']:.0%}", v["meaning"])
print(pd.crosstab(t["Species"], t["tag"]).to_string())
