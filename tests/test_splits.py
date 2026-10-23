from pathlib import Path

import pandas as pd
import pytest

SPLITS = Path("data/splits")
pytestmark = pytest.mark.skipif(not (SPLITS / "train.csv").exists(), reason="splits not built")


def _load():
    return {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "val", "test")}


def test_no_file_or_group_overlap():
    d = _load()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(d[a]["Filename"]) & set(d[b]["Filename"])
        assert not set(d[a]["group"]) & set(d[b]["group"])


def test_covers_everything_once():
    d = _load()
    all_files = pd.concat([d[s]["Filename"] for s in d])
    assert len(all_files) == 17509 and all_files.is_unique


def test_class_balance_per_split():
    d = _load()
    total = pd.concat(d.values())["Label"].value_counts()
    for s, lo, hi in (("train", 0.66, 0.74), ("val", 0.12, 0.18), ("test", 0.12, 0.18)):
        share = d[s]["Label"].value_counts() / total
        assert len(share) == 9
        assert share.between(lo, hi).all(), f"{s}: {share.round(3).to_dict()}"


def test_official_split_is_disjoint():
    parts = [pd.read_csv(SPLITS / f"official_{s}.csv")["Filename"] for s in ("train", "val", "test")]
    assert len(set(parts[0]) & set(parts[2])) == 0
