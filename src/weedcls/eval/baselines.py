"""M0: always predict the most common training class. A floor that shows why accuracy misleads."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weedcls.eval.metrics import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="reports/metrics/m0_majority_val.json")
    a = ap.parse_args()
    tr = pd.read_csv(Path(a.splits) / "train.csv")
    va = pd.read_csv(Path(a.splits) / "val.csv")
    majority = int(tr["Label"].mode()[0])
    m = compute_metrics(va["Label"].to_numpy(), np.full(len(va), majority))
    m["predicts_label"] = majority
    Path(a.out).write_text(json.dumps(m, indent=2))
    print({k: round(m[k], 4) for k in ("accuracy", "balanced_accuracy", "macro_f1", "weed_recall")})


if __name__ == "__main__":
    main()
