import numpy as np

from weedcls.eval.metrics import bootstrap_ci, compute_metrics


def test_majority_baseline_has_high_accuracy_low_macro_f1():
    y = np.array([8] * 52 + list(range(8)) * 6)  # 52 negatives, 48 weeds
    m = compute_metrics(y, np.full_like(y, 8))
    assert m["accuracy"] == 0.52
    assert m["macro_f1"] < 0.1
    assert m["weed_recall"] == 0.0 and m["false_alarm_rate"] == 0.0  # never raises an alarm


def test_perfect_prediction():
    y = np.arange(9).repeat(3)
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["weed_recall"] == 1.0 and m["false_alarm_rate"] == 0.0


def test_weed_confusion_between_species_still_counts_as_weed_found():
    y_true = np.array([0, 0, 8, 8])
    y_pred = np.array([1, 1, 8, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["weed_recall"] == 1.0 and m["false_alarm_rate"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 9, 600)
    pred = np.where(rng.random(600) < 0.7, y, rng.integers(0, 9, 600))
    groups = np.repeat(np.arange(30), 20)
    lo, hi = bootstrap_ci(y, pred, groups, n_boot=200)
    assert lo < compute_metrics(y, pred)["macro_f1"] < hi
