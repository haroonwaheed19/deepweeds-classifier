"""Temperature scaling and reliability diagrams. Fit on VALIDATION logits only."""
import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising NLL of softmax(logits / T). Doesn't change argmax/accuracy."""
    z, t = torch.from_numpy(logits).float(), torch.from_numpy(y).long()
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_plot(probs_before, probs_after, y, path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    edges = np.linspace(0, 1, n_bins + 1)
    for probs, label, color in ((probs_before, "before", "#c0504d"), (probs_after, "after T-scaling", "#4a7c59")):
        conf, pred = probs.max(1), probs.argmax(1)
        xs, ys = [], []
        for lo, hi in itertools.pairwise(edges):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys.append((pred[m] == y[m]).mean())
        ax.plot(xs, ys, "o-", label=f"{label} (ECE {expected_calibration_error(probs, y, n_bins):.3f})",
                color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    ax.set_title("Reliability (validation)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fit_negative_bias(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0,
                      negative: int = 8, grid=None) -> tuple[float, list[dict]]:
    """Add a constant to the Negative logit (after dividing by T) to rebalance weed vs no-weed.

    Weighted training makes models over-eager to call weeds; this undoes part of it without
    retraining. Picked by validation macro-F1. Returns (best bias, the whole curve for reporting).
    """
    from weedcls.eval.metrics import compute_metrics

    grid = np.linspace(-2.0, 4.0, 61) if grid is None else grid
    curve = []
    for b in grid:
        z = logits / temperature
        z = z.copy()
        z[:, negative] += b
        m = compute_metrics(y, z.argmax(1))
        curve.append({"bias": round(float(b), 3), "macro_f1": m["macro_f1"],
                      "weed_recall": m["weed_recall"], "false_alarm_rate": m["false_alarm_rate"]})
    best = max(curve, key=lambda r: r["macro_f1"])
    return float(best["bias"]), curve


def choose_uncertain_threshold(probs: np.ndarray, y: np.ndarray, target_acc: float = 0.85):
    """Smallest confidence cut-off whose retained (confident) predictions reach target_acc.

    Returns (threshold, coverage, accuracy_above). If the target is unreachable, returns the
    cut-off with the best accuracy among those keeping at least 20% of the data.
    """
    conf, correct = probs.max(1), (probs.argmax(1) == y)
    best = None
    for t in np.unique(np.round(conf, 3)):
        keep = conf >= t
        if keep.mean() < 0.2:
            break
        acc = correct[keep].mean()
        if acc >= target_acc:
            return float(t), float(keep.mean()), float(acc)
        if best is None or acc > best[2]:
            best = (float(t), float(keep.mean()), float(acc))
    return best
