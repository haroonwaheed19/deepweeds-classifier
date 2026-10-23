"""Session-grouped, class-stratified train/val/test split.

Why not a random split: frames from one capture session share lighting, background and colour
cast, and most sessions contain only 1-3 classes (see docs/experiments.md), so a random split
lets a model score by recognising the session. We group by (date, hour) instead. Whole dates
are impossible because Parkinsonia exists on only 2 dates.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from weedcls.data.inspect import load_labels
from weedcls.logging_utils import get_logger

log = get_logger("splits")

TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def session_groups(df: pd.DataFrame, dup_pairs: pd.DataFrame | None = None) -> pd.Series:
    """Group id per image: date + hour, with near-duplicate pairs merged into one group."""
    key = df["date"] + "-" + df["time"].str[:2]
    parent = {k: k for k in key.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if dup_pairs is not None and len(dup_pairs):
        fkey = dict(zip(df["Filename"], key, strict=True))
        for a, b in zip(dup_pairs["file_a"], dup_pairs["file_b"], strict=True):
            ra, rb = find(fkey[a]), find(fkey[b])
            if ra != rb:
                parent[rb] = ra
    return key.map(find)


def _loss(split_counts: np.ndarray, class_tot: np.ndarray, target: np.ndarray) -> float:
    """Squared deviation of every (split, class) share from its target share."""
    share = split_counts / class_tot[None, :]
    return float(((share - target[:, None]) ** 2).sum())


def make_split(df: pd.DataFrame, n_tries: int = 40, iters: int = 6000) -> tuple[pd.Series, int, float]:
    """Assign whole groups to splits by hill-climbing so every class lands near 70/15/15.

    Plain StratifiedGroupKFold was too lumpy here (a handful of groups hold 500+ images), e.g.
    it left Parkinsonia with 16 test images. With ~190 groups a direct search is cheap.
    """
    classes = sorted(df["Label"].unique())
    groups = sorted(df["group"].unique())
    gmat = (
        pd.crosstab(df["group"], df["Label"]).reindex(index=groups, columns=classes, fill_value=0).to_numpy()
    )
    class_tot = gmat.sum(0).astype(float)
    target = np.array([TARGET[s] for s in ("train", "val", "test")])
    n_g = len(groups)

    def counts(a):
        return np.stack([gmat[a == k].sum(0) for k in range(3)]).astype(float)

    best_a, best_seed, best_loss = None, -1, float("inf")
    for seed in range(n_tries):
        rng = np.random.default_rng(seed)
        a = rng.choice(3, size=n_g, p=target)
        cur = _loss(counts(a), class_tot, target)
        for _ in range(iters):
            b = a.copy()
            if rng.random() < 0.5:
                b[rng.integers(n_g)] = rng.integers(3)
            else:  # swap two groups' splits
                i, j = rng.integers(n_g, size=2)
                b[i], b[j] = a[j], a[i]
            new = _loss(counts(b), class_tot, target)
            if new <= cur:
                a, cur = b, new
        if cur < best_loss:
            best_a, best_seed, best_loss = a.copy(), seed, cur
    lookup = {g: ("train", "val", "test")[k] for g, k in zip(groups, best_a, strict=True)}
    return df["group"].map(lookup), best_seed, best_loss


def summarize(df: pd.DataFrame, assign: pd.Series, names: dict) -> dict:
    df = df.assign(split=assign)
    counts = pd.crosstab(df["Species"], df["split"])[["train", "val", "test"]]
    shared_dates = {
        f"{a}&{b}": len(set(df.loc[df.split == a, "date"]) & set(df.loc[df.split == b, "date"]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "per_split_counts": {s: int((df.split == s).sum()) for s in TARGET},
        "per_split_fraction": {s: round(float((df.split == s).mean()), 4) for s in TARGET},
        "class_counts": counts.to_dict(orient="index"),
        "min_class_fraction_in_val_or_test": round(
            float(pd.crosstab(df["Species"], df["split"], normalize="index")[["val", "test"]].min().min()), 4
        ),
        "n_groups": int(df["group"].nunique()),
        "groups_per_split": {s: int(df.loc[df.split == s, "group"].nunique()) for s in TARGET},
        "dates_shared_between_splits": shared_dates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--reports", default="reports/metrics")
    ap.add_argument("--tries", type=int, default=40)
    args = ap.parse_args()

    df = load_labels(Path(args.root) / "labels" / "labels.csv")
    dup_path = Path(args.features) / "near_dup_pairs.csv"
    dups = None
    if dup_path.exists():
        dups = pd.read_csv(dup_path)
        dups = dups[dups["dist"] <= 4]
    else:
        log.warning("no near_dup_pairs.csv, run weedcls.data.inspect first; grouping without it")
    df["group"] = session_groups(df, dups)

    assign, seed, score = make_split(df, args.tries)
    log.info("best restart %d, loss %.4f", seed, score)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in TARGET:
        part = df.loc[assign == s, ["Filename", "Label", "Species", "group"]]
        part.sort_values("Filename").to_csv(out / f"{s}.csv", index=False)
    # the paper's random 60/20/20 split, kept only to measure how much random splitting inflates scores
    for s in TARGET:
        src = Path(args.root) / "labels" / f"{s}_subset0.csv"
        shutil.copy(src, out / f"official_{s}.csv")

    names = df.groupby("Label")["Species"].first().to_dict()
    summ = summarize(df, assign, names)
    summ["seed"] = seed
    summ["search_loss"] = round(score, 5)
    summ["grouping"] = "date+hour, near-duplicate pairs (pHash<=4) merged"
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    (Path(args.reports) / "split_summary.json").write_text(json.dumps(summ, indent=2))
    log.info("%s", json.dumps({k: summ[k] for k in ("per_split_counts", "per_split_fraction", "dates_shared_between_splits")}))


if __name__ == "__main__":
    main()
