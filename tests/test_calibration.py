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
