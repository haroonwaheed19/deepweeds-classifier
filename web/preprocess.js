// Browser-side twin of weedcls.preprocess + weedcls.inference.predictor.
//
// The model was validated on pixels produced by Pillow's Image.resize(BILINEAR) (antialiased,
// 22-bit fixed point, horizontal pass then vertical pass, each rounded to uint8). A canvas
// drawImage() downscale uses a different filter, which shifts pixels slightly, so the resize is
// re-implemented here following Pillow's Resample.c. web/test/parity.test.mjs checks it against
// the Python output.

const PRECISION_BITS = 22;

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1.0);
  const support = 1.0 * filterScale; // bilinear support is 1.0
  const ss = 1.0 / filterScale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const v = Math.abs((x + xmin - center + 0.5) * ss);
      const w = v < 1.0 ? 1.0 - v : 0.0;
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const kv = ww === 0 ? k[x] : k[x] / ww;
      kk[xx * ksize + x] = Math.trunc(kv * (1 << PRECISION_BITS) + (kv < 0 ? -0.5 : 0.5));
    }
    bounds[2 * xx] = xmin;
    bounds[2 * xx + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

function clip8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** rgba: Uint8ClampedArray/Uint8Array of w*h*4. Returns Uint8Array of size*size*3 (RGB). */
export function resizeBilinearPil(rgba, w, h, size) {
  let cur = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
    cur[j] = rgba[i];
    cur[j + 1] = rgba[i + 1];
    cur[j + 2] = rgba[i + 2];
  }
  let cw = w;
  let ch = h;
  const half = 1 << (PRECISION_BITS - 1);

  if (cw !== size) {
    const { bounds, kk, ksize } = coefficients(cw, size);
    const out = new Uint8Array(size * ch * 3);
    for (let y = 0; y < ch; y++) {
      for (let xx = 0; xx < size; xx++) {
        const xmin = bounds[2 * xx];
        const n = bounds[2 * xx + 1];
        for (let c = 0; c < 3; c++) {
          let sum = half;
          for (let x = 0; x < n; x++) sum += cur[(y * cw + xmin + x) * 3 + c] * kk[xx * ksize + x];
          out[(y * size + xx) * 3 + c] = clip8(sum >> PRECISION_BITS);
        }
      }
    }
    cur = out;
    cw = size;
  }
  if (ch !== size) {
    const { bounds, kk, ksize } = coefficients(ch, size);
    const out = new Uint8Array(cw * size * 3);
    for (let yy = 0; yy < size; yy++) {
      const ymin = bounds[2 * yy];
      const n = bounds[2 * yy + 1];
      for (let x = 0; x < cw; x++) {
        for (let c = 0; c < 3; c++) {
          let sum = half;
          for (let y = 0; y < n; y++) sum += cur[((ymin + y) * cw + x) * 3 + c] * kk[yy * ksize + y];
          out[(yy * cw + x) * 3 + c] = clip8(sum >> PRECISION_BITS);
        }
      }
    }
    cur = out;
    ch = size;
  }
  return cur;
}

/** RGBA pixels -> normalised float32 CHW tensor data, same as preprocess_pil. */
export function toTensorData(rgba, w, h, size, mean, std) {
  const rgb = resizeBilinearPil(rgba, w, h, size);
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (rgb[i * 3 + c] / 255.0 - mean[c]) / std[c];
    }
  }
  return out;
}

/** logits -> result object with the same fields as the REST API's /predict response. */
export function postprocess(logits, card) {
  const T = card.temperature ?? 1.0;
  const z = Array.from(logits, (v) => v / T);
  z[card.negative_index] += card.negative_logit_bias ?? 0.0;
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => v / s);
  const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
  const top = order[0];
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  return {
    label: card.classes[top],
    confidence: r4(p[top]),
    is_weed: top !== card.negative_index,
    weed_probability: r4(1 - p[card.negative_index]),
    uncertain: p[top] < (card.uncertain_threshold ?? 0),
    top_k: order.slice(0, 3).map((i) => ({ label: card.classes[i], probability: r4(p[i]) })),
    model_version: card.model_version,
  };
}
