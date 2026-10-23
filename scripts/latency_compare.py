"""Single-image CPU latency: PyTorch vs ONNX Runtime (forward only), plus the full Predictor path."""
import json
import time
from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image

from weedcls.eval.evaluate import load_torch
from weedcls.inference.predictor import Predictor


def timeit(fn, n=60):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return round((time.perf_counter() - t0) / n * 1000, 1)


torch.set_num_threads(2)  # same budget as the service default (HF free tier is 2 vCPU)
model, size, _, _ = load_torch(Path("runs/m3_mobilenet_ft"))
x = torch.randn(1, 3, size, size)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
sess = ort.InferenceSession("artifacts/model.onnx", opts, providers=["CPUExecutionProvider"])
xn = x.numpy()
img = Image.open("data/raw/images/20170130-102111-0.jpg")
pred = Predictor("artifacts/model.onnx", "artifacts/model_card.json", threads=2)
with torch.no_grad():
    res = {
        "threads": 2,
        "torch_forward_ms": timeit(lambda: model(x)),
        "onnx_forward_ms": timeit(lambda: sess.run(None, {"input": xn})),
        "predictor_end_to_end_ms": timeit(lambda: pred.predict([img])),
    }
Path("reports/metrics/latency.json").write_text(json.dumps(res, indent=2))
print(res)
