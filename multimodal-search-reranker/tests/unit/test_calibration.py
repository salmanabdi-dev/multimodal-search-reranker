import numpy as np
import pytest

from mmsearch.search.calibration import Platt, fit_platt, log_loss, sigmoid


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(0) == 0.5
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0


def test_fit_platt_recovers_known_parameters():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.1, 0.4, 20_000)
    true = Platt(30.0, -7.5)
    y = (rng.random(s.size) < np.array([true(x) for x in s])).astype(int)
    fit = fit_platt(s, y)
    assert fit.a == pytest.approx(30.0, rel=0.1)
    assert fit.b == pytest.approx(-7.5, rel=0.1)


def test_fitted_calibration_beats_a_bad_one():
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(0.30, 0.02, 500), rng.normal(0.20, 0.02, 4500)])
    y = np.array([1] * 500 + [0] * 4500)
    fit = fit_platt(s, y)
    bad = Platt(1.0, 0.0)
    assert log_loss([fit(x) for x in s], y) < log_loss([bad(x) for x in s], y)
    assert fit.a > 0  # higher similarity -> higher relevance


def test_fit_needs_both_classes():
    with pytest.raises(ValueError):
        fit_platt([0.1, 0.2], [1, 1])
