import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

NEGATIVE = 8


def compute_metrics(y_true, y_pred, num_classes: int = 9, negative: int = NEGATIVE) -> dict:
    """Macro-F1 is the headline: 52% of images are Negative, so accuracy flatters a lazy model.

    weed_recall / false_alarm_rate collapse the problem to 'is there any weed here', which is
    what a sprayer cares about: a miss leaves a weed, a false alarm wastes herbicide.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    # average only over classes that exist in y_true: a bootstrap resample can miss a rare
    # class entirely, and counting that as F1 = 0 would drag the interval down
    present = [c for c in labels if s[c] > 0]
    true_weed, pred_weed = y_true != negative, y_pred != negative
    tp = int((true_weed & pred_weed).sum())
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(r[present])),
        "macro_f1": float(np.mean(f[present])),
        "per_class": {
            str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c])}
            for c in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "weed_recall": tp / max(int(true_weed.sum()), 1),
        "weed_precision": tp / max(int(pred_weed.sum()), 1),
        "false_alarm_rate": float((~true_weed & pred_weed).sum() / max(int((~true_weed).sum()), 1)),
    }


def bootstrap_ci(y_true, y_pred, groups, metric: str = "macro_f1", n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Cluster bootstrap: resample whole capture groups, since frames within a group are
    near-copies of each other and resampling single images would give CIs that are too tight."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        vals.append(compute_metrics(y_true[idx], y_pred[idx])[metric])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, groups, metric: str = "macro_f1",
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """CI on metric(A) - metric(B) using the same resampled groups for both models."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        diffs.append(compute_metrics(y_true[idx], pred_a[idx])[metric]
                     - compute_metrics(y_true[idx], pred_b[idx])[metric])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(diffs)), float(lo), float(hi)
