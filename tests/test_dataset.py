import numpy as np
import pandas as pd
import pytest
from PIL import Image

from weedcls.data.dataset import WeedDataset, class_weights


def test_missing_files_are_skipped(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    csv = tmp_path / "s.csv"
    pd.DataFrame({"Filename": ["a.jpg", "gone.jpg"], "Label": [0, 1]}).to_csv(csv, index=False)
    ds = WeedDataset(csv, tmp_path)
    assert len(ds) == 1


def test_all_missing_raises(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame({"Filename": ["x.jpg"], "Label": [0]}).to_csv(csv, index=False)
    with pytest.raises(FileNotFoundError):
        WeedDataset(csv, tmp_path)


def test_class_weights_favour_rare_classes_and_average_one():
    y = np.array([8] * 90 + [0] * 10)
    w = class_weights(y, num_classes=9)
    assert w[0] > w[8]
    assert abs(float((w[y]).mean()) - 1.0) < 1e-5
