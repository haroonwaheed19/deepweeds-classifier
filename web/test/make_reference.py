"""Write the fixtures and expected values that web/test/parity.test.mjs compares against.

Run from the repo root:  python web/test/make_reference.py
Fixtures are lossless PNGs so Python and Node decode identical pixels.
"""
import json
from pathlib import Path

from PIL import Image

from weedcls.inference.predictor import Predictor
from weedcls.preprocess import preprocess_pil

OUT = Path("web/test/fixtures")
OUT.mkdir(parents=True, exist_ok=True)
STRIDE = 37

pred = Predictor("artifacts/model.onnx", "artifacts/model_card.json")
card = pred.card

sources = {
    "lantana": Image.open("service/samples/lantana.jpg").convert("RGB"),
    "negative": Image.open("service/samples/negative.jpg").convert("RGB"),
    "siam_weed": Image.open("service/samples/siam_weed.jpg").convert("RGB"),
}
base = sources["lantana"]
sources["lantana_wide_400x300"] = base.crop((0, 0, 256, 192)).resize((400, 300), Image.BICUBIC)
sources["negative_small_160"] = sources["negative"].resize((160, 160), Image.BICUBIC)
sources["siam_big_900x700"] = sources["siam_weed"].resize((900, 700), Image.BICUBIC)

ref = {"stride": STRIDE, "cases": {}}
for name, img in sources.items():
    img.save(OUT / f"{name}.png")
    tensor = preprocess_pil(img, card["input_size"], card["mean"], card["std"])
    logits = pred.logits([img])[0]
    ref["cases"][name] = {
        "width": img.width, "height": img.height,
        "tensor_samples": [float(v) for v in tensor.reshape(-1)[::STRIDE]],
        "logits": [float(v) for v in logits],
        "result": {k: v for k, v in pred.predict([img])[0].items() if k != "latency_ms"},
    }
    print(name, img.size, "->", ref["cases"][name]["result"]["label"])
Path("web/test/reference.json").write_text(json.dumps(ref))
print("tensor sample count per case:", len(ref["cases"]["lantana"]["tensor_samples"]),
      "| total bytes:", sum(p.stat().st_size for p in OUT.glob("*.png")))
