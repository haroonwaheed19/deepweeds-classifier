// Checks that the browser code path (web/preprocess.js) reproduces the Python pipeline.
//   cd web/test && npm install && npm test
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { PNG } from "pngjs";
import { postprocess, toTensorData } from "../preprocess.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "reference.json"), "utf8"));
const card = JSON.parse(readFileSync(join(here, "..", "..", "artifacts", "model_card.json"), "utf8"));
const session = await ort.InferenceSession.create(join(here, "..", "..", "artifacts", "model.onnx"));

for (const [name, want] of Object.entries(ref.cases)) {
  const png = PNG.sync.read(readFileSync(join(here, "fixtures", `${name}.png`)));
  assert.equal(png.width, want.width);
  const tensor = toTensorData(png.data, png.width, png.height, card.input_size, card.mean, card.std);

  test(`${name}: preprocessed tensor matches Python`, () => {
    let worst = 0;
    want.tensor_samples.forEach((v, i) => {
      worst = Math.max(worst, Math.abs(tensor[i * ref.stride] - v));
    });
    assert.ok(worst < 1e-5, `max abs diff ${worst}`);
  });

  test(`${name}: logits and final result match Python`, async () => {
    const out = await session.run({
      input: new ort.Tensor("float32", tensor, [1, 3, card.input_size, card.input_size]),
    });
    const logits = Array.from(out.logits.data);
    const worst = Math.max(...logits.map((v, i) => Math.abs(v - want.logits[i])));
    assert.ok(worst < 1e-3, `max logit diff ${worst}`);
    const got = postprocess(logits, card);
    assert.equal(got.label, want.result.label);
    assert.equal(got.is_weed, want.result.is_weed);
    assert.equal(got.uncertain, want.result.uncertain);
    assert.ok(Math.abs(got.confidence - want.result.confidence) < 2e-3);
  });
}
