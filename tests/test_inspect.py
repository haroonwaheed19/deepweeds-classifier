import numpy as np
import pandas as pd

from weedcls.data.inspect import load_labels, near_duplicate_pairs


def test_load_labels_parses_filename(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame(
        {"Filename": ["20170610-123859-1.jpg"], "Label": [1], "Species": ["Lantana"]}
    ).to_csv(p, index=False)
    df = load_labels(p)
    assert df.loc[0, "date"] == "20170610"
    assert df.loc[0, "time"] == "123859"
    assert df.loc[0, "cam"] == "1"


def test_near_duplicate_pairs_hamming():
    h = np.array([0b0000, 0b0001, 0b1111_0000], dtype=np.uint64)
    pairs = near_duplicate_pairs(h, max_dist=1)
    assert pairs == [(0, 1, 1)]
