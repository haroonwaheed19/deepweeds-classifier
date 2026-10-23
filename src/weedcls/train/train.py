"""Config-driven trainer for the small CNN and fine-tuned backbones.

    python -m weedcls.train.train --config configs/small_cnn.yaml --override train.lr=1e-3

Each run lives in runs/<date>_<name>/ with the resolved config, git hash, per-epoch metrics,
best.pt / last.pt and the validation logits of the best epoch. Re-run with --resume to continue.
"""
import argparse
import csv
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from weedcls.config import load_config
from weedcls.data.dataset import CLASS_NAMES, WeedDataset, class_weights, make_loader, make_sampler
from weedcls.data.transforms import build_eval_transform, build_train_transform
from weedcls.eval.metrics import compute_metrics
from weedcls.logging_utils import get_logger
from weedcls.models.registry import (
    build_model,
    freeze_for_finetune,
    head_parameters,
    set_train_mode,
)
from weedcls.preprocess import IMAGENET_MEAN, IMAGENET_STD
from weedcls.seed import set_seed
from weedcls.train.engine import predict_logits

COLUMNS = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_acc", "val_bal_acc",
           "val_weed_recall", "val_false_alarm", "lr", "seconds"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001  not a git checkout, or git missing
        return "unknown"


def _norm(cfg: dict) -> tuple:
    if cfg["norm"] == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    stats = json.loads(Path(cfg["_data"]["paths"]["norm_stats"]).read_text())
    return tuple(stats["mean"]), tuple(stats["std"])


def _split_csv(cfg: dict, which: str) -> Path:
    prefix = "official_" if cfg["_data"].get("split", "grouped") == "official" else ""
    return Path(cfg["_data"]["paths"]["splits"]) / f"{prefix}{which}.csv"


def build_data(cfg: dict):
    size, (mean, std) = cfg["model"]["img_size"], _norm(cfg)
    dpaths, tcfg = cfg["_data"]["paths"], cfg["train"]
    train_csv = _split_csv(cfg, "train")
    frac = tcfg.get("subset_fraction", 1.0)
    if frac < 1.0:  # stratified subsample, used for quick hyperparameter runs / smoke tests
        df = pd.read_csv(train_csv)
        df = df.groupby("Label", group_keys=False).sample(frac=frac, random_state=cfg["seed"])
        train_csv = Path(cfg["_run_dir"]) / "train_subset.csv"
        df.to_csv(train_csv, index=False)
    train_ds = WeedDataset(train_csv, dpaths["images"],
                           build_train_transform(size, mean, std, cfg["_data"].get("augment")))
    val_ds = WeedDataset(_split_csv(cfg, "val"), dpaths["images"], build_eval_transform(size, mean, std))
    nw = cfg["_data"].get("num_workers", 2)
    sampler = make_sampler(train_ds.labels) if cfg["imbalance"] == "sampler" else None
    train_loader = make_loader(train_ds, tcfg["batch_size"], True, nw, sampler)
    val_loader = make_loader(val_ds, tcfg["batch_size"] * 2, False, nw)
    return train_ds, train_loader, val_loader


def build_optimizer(model, cfg):
    t, name = cfg["train"], cfg["model"]["name"]
    if name == "small_cnn":
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": t["lr"]}]
    else:
        head_ids = {id(p) for p in head_parameters(model, name)}
        body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": list(head_parameters(model, name)), "lr": t["lr"]}]
        if body:
            groups.append({"params": body, "lr": t["lr"] * t.get("backbone_lr_mult", 0.1)})
    return torch.optim.AdamW(groups, weight_decay=t.get("weight_decay", 1e-4))


def run(cfg: dict, resume: bool = False) -> Path:
    t = cfg["train"]
    set_seed(cfg["seed"])
    torch.set_num_threads(t.get("threads", 3))
    run_dir = Path(cfg["_run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"train.{run_dir.name}", run_dir / "train.log")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "_data"}))
    (run_dir / "git_hash.txt").write_text(_git_hash())

    train_ds, train_loader, val_loader = build_data(cfg)
    log.info("train %d imgs, val %d imgs, split=%s", len(train_ds), len(val_loader.dataset),
             cfg["_data"].get("split", "grouped"))

    model = build_model(cfg["model"], len(CLASS_NAMES))
    mname = cfg["model"]["name"]
    unfreeze_final = cfg["model"].get("unfreeze_last", 0)
    warmup = t.get("warmup_head_epochs", 0)
    if mname != "small_cnn":
        freeze_for_finetune(model, mname, 0 if warmup else unfreeze_final)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model %s, trainable params %.2fM", mname, n_train / 1e6)

    weight = class_weights(train_ds.labels) if cfg["imbalance"] == "weights" else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])

    start, best_f1, bad = 0, -1.0, 0
    metrics_path = run_dir / "metrics.csv"
    if resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_f1, bad = ck["epoch"] + 1, ck["best_f1"], ck["bad"]
        log.info("resumed from epoch %d (best val macro-F1 %.4f)", start, best_f1)
    elif metrics_path.exists():
        metrics_path.unlink()
    if not metrics_path.exists():
        metrics_path.write_text(",".join(COLUMNS) + "\n")

    for epoch in range(start, t["epochs"]):
        if warmup and epoch == warmup and mname != "small_cnn":
            freeze_for_finetune(model, mname, unfreeze_final)
            opt = build_optimizer(model, cfg)  # new param groups now that backbone blocks train
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] - epoch)
            log.info("epoch %d: unfroze last %d backbone blocks", epoch, unfreeze_final)
        t0 = time.time()
        set_train_mode(model)
        total, seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
            if i % 40 == 0:
                log.info("epoch %d batch %d/%d loss %.4f", epoch, i, len(train_loader), loss.item())
        sched.step()

        logits, ys = predict_logits(model, val_loader)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(ys)))
        m = compute_metrics(ys, logits.argmax(1))
        row = [epoch, total / seen, val_loss, m["macro_f1"], m["accuracy"], m["balanced_accuracy"],
               m["weed_recall"], m["false_alarm_rate"], opt.param_groups[0]["lr"], time.time() - t0]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([round(v, 5) if isinstance(v, float) else v for v in row])
        log.info("epoch %d | train loss %.4f | val loss %.4f | macro-F1 %.4f | acc %.4f | %.0fs",
                 epoch, row[1], val_loss, m["macro_f1"], m["accuracy"], row[-1])

        improved = m["macro_f1"] > best_f1
        if improved:
            best_f1, bad = m["macro_f1"], 0
            torch.save({"model": model.state_dict(), "cfg": {k: v for k, v in cfg.items() if k != "_data"},
                        "epoch": epoch, "val_macro_f1": best_f1}, run_dir / "best.pt")
            np.savez(run_dir / "val_logits.npz", logits=logits, y=ys)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "bad": bad}, run_dir / "last.pt")
        if bad >= t.get("patience", 5):
            log.info("early stopping: no val macro-F1 gain for %d epochs", bad)
            break

    (run_dir / "summary.json").write_text(json.dumps({"best_val_macro_f1": best_f1}, indent=2))
    log.info("done. best val macro-F1 %.4f", best_f1)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    cfg["_data"] = load_config(cfg.get("data_config", "configs/data.yaml"))
    name = args.run_name or f"{date.today():%Y%m%d}_{cfg['name']}"
    cfg["_run_dir"] = str(Path(args.runs_dir) / name)
    run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
