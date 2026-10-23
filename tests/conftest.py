import json

import pytest
import torch
from torch import nn

from weedcls.inference.export_onnx import export, parity


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory):
    """A tiny random-weight ONNX model + card, so serving tests don't need training outputs."""
    d = tmp_path_factory.mktemp("artifacts")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(4, 9)).eval()
    export(model, 64, d / "model.onnx")
    parity(model, d / "model.onnx", 64)
    classes = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
               "Rubber vine", "Siam weed", "Snake weed", "Negative"]
    card = {"model_version": "test", "input_size": 64, "mean": [0.5] * 3, "std": [0.25] * 3,
            "classes": classes, "negative_index": 8, "temperature": 1.0, "uncertain_threshold": 0.99}
    (d / "model_card.json").write_text(json.dumps(card))
    return d
