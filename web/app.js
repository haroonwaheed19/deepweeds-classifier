import { postprocess, toTensorData } from "./preprocess.js";

const MAX_BYTES = 12 * 1024 * 1024;
const MIN_SIDE = 32;
const MAX_SIDE = 6000;
const TYPES = ["image/jpeg", "image/png", "image/webp"];

const $ = (id) => document.getElementById(id);
let session = null;
let card = null;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").className = isError ? "warn" : "";
}

async function init() {
  try {
    card = await (await fetch("model_card.json")).json();
    ort.env.wasm.numThreads = 1; // multi-threaded wasm needs cross-origin isolation, unavailable in an iframe
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    setStatus("Model ready.");
    $("file").disabled = false;
    document.querySelectorAll(".sample").forEach((b) => (b.disabled = false));
  } catch (e) {
    setStatus(`Could not load the model: ${e.message}`, true);
    throw e;
  }
}

async function classify(blob) {
  if (!TYPES.includes(blob.type)) throw new Error(`unsupported type '${blob.type}', use JPEG, PNG or WebP`);
  if (blob.size > MAX_BYTES) throw new Error("file is larger than 12 MB");
  // imageOrientation "none": the Python pipeline does not apply EXIF rotation either
  const bmp = await createImageBitmap(blob, { imageOrientation: "none" });
  const { width: w, height: h } = bmp;
  if (Math.min(w, h) < MIN_SIDE || Math.max(w, h) > MAX_SIDE) {
    throw new Error(`image sides must be between ${MIN_SIDE} and ${MAX_SIDE}px, got ${w}x${h}`);
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;

  const t0 = performance.now();
  const data = toTensorData(rgba, w, h, card.input_size, card.mean, card.std);
  const input = new ort.Tensor("float32", data, [1, 3, card.input_size, card.input_size]);
  const out = await session.run({ input });
  const result = postprocess(Array.from(out.logits.data), card);
  result.latency_ms = Math.round(performance.now() - t0);
  return result;
}

async function run(blob) {
  $("verdict").textContent = "";
  $("json").hidden = true;
  try {
    setStatus("Running...");
    const url = URL.createObjectURL(blob);
    $("preview").src = url;
    $("preview").style.display = "block";
    const r = await classify(blob);
    $("verdict").innerHTML =
      `${r.label} (${(r.confidence * 100).toFixed(1)}%)` +
      (r.uncertain ? ' <span class="warn">- low confidence</span>' : "");
    $("json").textContent = JSON.stringify(r, null, 2);
    $("json").hidden = false;
    setStatus(`Done in ${r.latency_ms} ms.`);
    document.body.dataset.result = JSON.stringify(r);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
    document.body.dataset.error = e.message;
  }
}

$("file").addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  if (f) run(f);
});
document.querySelectorAll(".sample").forEach((b) =>
  b.addEventListener("click", async () => run(await (await fetch(`samples/${b.dataset.name}.jpg`)).blob())),
);

await init();
// ?sample=lantana runs a sample on load; used for automated browser checks
const auto = new URLSearchParams(location.search).get("sample");
if (auto) await run(await (await fetch(`samples/${auto}.jpg`)).blob());
document.body.dataset.ready = "1";
