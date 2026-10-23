"""ONNX Runtime predictor. Deliberately torch-free so the serving image stays small."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from weedcls.preprocess import preprocess_pil


def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Predictor:
    def __init__(self, model_path: str | Path, card_path: str | Path, threads: int = 2):
        card = json.loads(Path(card_path).read_text())
        self.card = card
        self.classes: list[str] = card["classes"]
        self.negative = card["negative_index"]
        self.size = card["input_size"]
        self.mean, self.std = card["mean"], card["std"]
        self.temperature = card.get("temperature", 1.0)
        self.uncertain_below = card.get("uncertain_threshold", 0.0)
        # fitted on validation logits: z/T first, then this offset on the Negative logit
        self.negative_bias = card.get("negative_logit_bias", 0.0)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([preprocess_pil(im, self.size, self.mean, self.std) for im in images])
        return self.session.run(None, {self.input_name: batch})[0]

    def predict(self, images: list[Image.Image], top_k: int = 3) -> list[dict]:
        t0 = time.perf_counter()
        z = self.logits(images) / self.temperature
        z[:, self.negative] += self.negative_bias
        probs = _softmax(z, 1.0)
        per_image_ms = (time.perf_counter() - t0) * 1000 / len(images)
        out = []
        for p in probs:
            order = np.argsort(-p)
            label = int(order[0])
            out.append({
                "label": self.classes[label],
                "confidence": round(float(p[label]), 4),
                "is_weed": label != self.negative,
                "weed_probability": round(float(1.0 - p[self.negative]), 4),
                "uncertain": bool(p[label] < self.uncertain_below),
                "top_k": [{"label": self.classes[i], "probability": round(float(p[i]), 4)}
                          for i in order[:top_k]],
                "latency_ms": round(per_image_ms, 1),
            })
        return out
