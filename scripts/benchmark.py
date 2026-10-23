"""Measure CPU throughput of each candidate so model/resolution choices rest on numbers."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from weedcls.logging_utils import get_logger
from weedcls.models.backbones import build_backbone
from weedcls.models.small_cnn import SmallCNN

log = get_logger("benchmark")


def _time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench(model: nn.Module, size: int, batch: int, iters: int) -> dict:
    x = torch.randn(batch, 3, size, size)
    model.eval()

    def fwd():
        with torch.no_grad():
            model(x)

    t_fwd = _time(fwd, 1, iters)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad()
        model(x).float().mean().backward()
        opt.step()

    t_train = _time(step, 1, max(2, iters // 2))
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "forward_img_per_s": round(batch / t_fwd, 1),
        "train_img_per_s": round(batch / t_train, 1),
        "params_m": round(n_params / 1e6, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", default="reports/metrics/benchmark.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cases = [("small_cnn", 128), ("small_cnn", 224)]
    for name in ("mobilenet_v3_large", "efficientnet_b0", "resnet18"):
        cases += [(name, 160), (name, 224)]

    results = []
    for name, size in cases:
        model = SmallCNN(9) if name == "small_cnn" else build_backbone(name, pretrained=False)
        r = {"model": name, "input": size, **bench(model, size, args.batch, args.iters)}
        # 12.2k train images in the grouped split's training portion (approx. 70% of 17,509)
        r["est_min_per_train_epoch"] = round(12256 / r["train_img_per_s"] / 60, 1)
        log.info("%s", r)
        results.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"threads": args.threads, "results": results}, indent=2))
    log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
