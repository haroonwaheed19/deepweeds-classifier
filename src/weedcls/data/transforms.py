import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from weedcls.preprocess import preprocess_pil


class EvalTransform:
    """Deterministic. Thin wrapper so train/eval/serve share weedcls.preprocess."""

    def __init__(self, size: int, mean, std):
        self.size, self.mean, self.std = size, mean, std

    def __call__(self, img):
        return torch.from_numpy(preprocess_pil(img, self.size, self.mean, self.std))


class RandomRot90:
    """Top-down camera: orientation carries no information, so any multiple of 90 degrees is fair."""

    def __call__(self, img):
        return F.rotate(img, random.choice((0, 90, 180, 270)))


def build_train_transform(size: int, mean, std, aug: dict | None = None) -> T.Compose:
    aug = aug or {}
    crop_lo = aug.get("crop_scale_min", 0.7)
    cj = aug.get("color_jitter", {"brightness": 0.3, "contrast": 0.3, "saturation": 0.1, "hue": 0.02})
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(crop_lo, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            RandomRot90(),
            T.ColorJitter(**cj),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_eval_transform(size: int, mean, std) -> EvalTransform:
    return EvalTransform(size, mean, std)
