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
