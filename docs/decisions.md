# Decisions

Running log of choices and the evidence behind them. Newest at the bottom.

## 2026-10-23: compute budget (from `scripts/benchmark.py`)

Machine: i3-1005G1 (2 cores / 4 threads), 7.7 GB RAM, no GPU. Torch 2.14 CPU, 4 threads, batch 32,
random tensors. Epoch estimates assume ~12.3k training images (70% of 17,509).

| model | input | forward img/s | train img/s | params (M) | est. min / train epoch |
|---|---|---|---|---|---|
| small_cnn | 128 | 101.9 | 52.0 | 0.32 | 3.9 |
| small_cnn | 224 | 36.8 | 16.4 | 0.32 | 12.5 |
| mobilenet_v3_large | 160 | 119.3 | 23.4 | 5.48 | 8.7 |
| mobilenet_v3_large | 224 | 58.0 | 13.1 | 5.48 | 15.6 |
| efficientnet_b0 | 160 | 71.7 | 14.0 | 5.29 | 14.6 |
| efficientnet_b0 | 224 | 38.5 | 5.6 | 5.29 | 36.5 |
| resnet18 | 160 | 48.1 | 16.0 | 11.69 | 12.8 |
| resnet18 | 224 | 27.8 | 9.4 | 11.69 | 21.7 |

What this means:

- **Feature caching is cheap.** One forward pass over all 17.5k images is ~5 min for MobileNetV3-L
  at 224 and ~8 min for EfficientNet-B0. So the linear-probe sweep (P3) costs almost nothing
  once embeddings are cached, and that is where most hyperparameter tuning will happen.
- **EfficientNet-B0 is out for full fine-tuning.** 36 min/epoch at 224 breaks the 25 min gate.
  It stays in as a frozen-feature candidate only.
- **MobileNetV3-L is the fine-tuning default.** Training with all layers unfrozen is 15.6 min/epoch
  at 224 and 8.7 at 160. Freezing the early blocks means no backward pass through them, so it
  should be noticeably cheaper than these all-layer numbers. To be re-measured in P4.
- **Small CNN at 128px** (3.9 min/epoch, ~80 min for 20 epochs) is the from-scratch baseline. 224
  is 3x slower with no reason to expect better results from a model this small.
- ResNet-18 is slower than MobileNetV3 here despite lower FLOPs claims elsewhere; with 2 cores
  memory-bound layers dominate. Kept as a frozen-feature candidate only.

Caveat: the first timed iteration after warmup can still be noisy on a laptop CPU (thermal
throttling). Treat these as +/-20%. Re-run if a decision depends on a close call.
