import numpy as np

from weedcls.eval.calibration import expected_calibration_error, fit_temperature
from weedcls.train.engine import softmax_np


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 1.5          # informative but noisy
    overconfident = logits * 4.0            # same argmax, sharpened
    t = fit_temperature(overconfident, y)
    assert t > 2.0
    before = expected_calibration_error(softmax_np(overconfident), y)
    after = expected_calibration_error(softmax_np(overconfident, t), y)
    assert after < before
    assert (softmax_np(overconfident, t).argmax(1) == overconfident.argmax(1)).all()


def test_negative_bias_reduces_false_alarms_on_biased_model():
    from weedcls.eval.calibration import fit_negative_bias
    from weedcls.eval.metrics import compute_metrics

    rng = np.random.default_rng(1)
    n = 3000
    y = np.where(rng.random(n) < 0.5, 8, rng.integers(0, 8, n))
    z = rng.normal(size=(n, 9))
    z[np.arange(n), y] += 2.0
    z[:, 8] -= 1.5            # model that under-predicts Negative
    b, curve = fit_negative_bias(z, y)
    assert b > 0
    zb = z.copy()
    zb[:, 8] += b
    assert compute_metrics(y, zb.argmax(1))["macro_f1"] > compute_metrics(y, z.argmax(1))["macro_f1"]
    assert len(curve) == 61


def test_uncertain_threshold_hits_target_accuracy():
    from weedcls.eval.calibration import choose_uncertain_threshold

    rng = np.random.default_rng(2)
    conf = rng.uniform(0.3, 1.0, 2000)
    correct = rng.random(2000) < conf   # perfectly calibrated
    probs = np.full((2000, 3), 0.0)
    probs[np.arange(2000), 0] = conf
    probs[:, 1:] = ((1 - conf) / 2)[:, None]
    y = np.where(correct, 0, 1)
    t, cov, acc = choose_uncertain_threshold(probs, y, target_acc=0.85)
    assert acc >= 0.85 and 0.2 <= cov < 1.0 and 0.3 <= t <= 1.0
