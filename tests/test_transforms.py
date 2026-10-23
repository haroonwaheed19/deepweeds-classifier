import numpy as np
import torch
from PIL import Image

from weedcls.data.transforms import build_eval_transform, build_train_transform
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_pil


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))


def test_eval_is_deterministic_and_matches_preprocess():
    t = build_eval_transform(160, IMAGENET_MEAN, IMAGENET_STD)
    a, b = t(_img()), t(_img())
    assert a.shape == (3, 160, 160) and a.dtype == torch.float32
    assert torch.equal(a, b)
    ref = preprocess_pil(_img(), 160, IMAGENET_MEAN, IMAGENET_STD)
    assert np.allclose(a.numpy(), ref)


def test_train_transform_shape_and_randomness():
    t = build_train_transform(128, IMAGENET_MEAN, IMAGENET_STD)
    x = [t(_img()) for _ in range(4)]
    assert all(v.shape == (3, 128, 128) for v in x)
    assert not torch.equal(x[0], x[1])


def test_preprocess_handles_grayscale_and_rgba():
    g = Image.new("L", (100, 80), 128)
    rgba = Image.new("RGBA", (300, 300), (10, 20, 30, 255))
    for im in (g, rgba):
        assert preprocess_pil(im, 64, IMAGENET_MEAN, IMAGENET_STD).shape == (3, 64, 64)
