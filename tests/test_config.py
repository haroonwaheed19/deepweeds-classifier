import pytest

from weedcls.config import load_config


def test_override_and_missing(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\n  epochs: 5\n")
    cfg = load_config(p, ["train.lr=3e-4", "train.epochs=2"])
    assert cfg["train"]["lr"] == 3e-4 and cfg["train"]["epochs"] == 2
    with pytest.raises(KeyError):
        load_config(p, ["train.nope=1"])
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
