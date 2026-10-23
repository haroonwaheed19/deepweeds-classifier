"""Eval-time preprocessing with PIL + numpy only (no torch).

The training code and the ONNX service both go through preprocess_pil, so the model sees
exactly the same pixels in both places.
"""
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_pil(img: Image.Image, size: int, mean, std) -> np.ndarray:
    """RGB -> resize to size x size (whole frame kept, no crop) -> normalise -> float32 CHW."""
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))
