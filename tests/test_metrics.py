"""
Tests for comprsk.metrics (brier_score, integrated_brier_score, concordance).
"""

import numpy as np
import pytest

from pycrr import FineGrayModel
from pycrr.metrics import brier_score, concordance, integrated_brier_score
from conftest import simulate_competing_risks


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_returns_tuple(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 20)
        result = brier_score(model, X, time, event, eval_times)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shapes(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 20)
        t_out, bs = brier_score(model, X, time, event, eval_times)
        assert t_out.shape == (20,)
        assert bs.shape == (20,)

    def test_eval_times_passthrough(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.array([5.0, 10.0, 20.0])
        t_out, _ = brier_score(model, X, time, event, eval_times)
        assert np.allclose(t_out, eval_times)

    def test_bs_nonnegative(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 15)
        _, bs = brier_score(model, X, time, event, eval_times)
        assert np.all(bs >= 0)

    def test_bs_below_null(self, fitted_model_large):
        """A fitted model should beat the null (0.25) on average."""
        model, X, time, event = fitted_model_large
        eval_times = np.linspace(2, time.max() * 0.5, 20)
        _, bs = brier_score(model, X, time, event, eval_times)
        assert np.mean(bs) < 0.25

    def test_bs_finite(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 15)
        _, bs = brier_score(model, X, time, event, eval_times)
        assert np.all(np.isfinite(bs))

    def test_cause_2(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 10)
        t_out, bs = brier_score(model, X, time, event, eval_times, cause=2)
        assert bs.shape == (10,)
        assert np.all(bs >= 0)


# ---------------------------------------------------------------------------
# Integrated Brier score
# ---------------------------------------------------------------------------

class TestIntegratedBrierScore:
    def test_returns_float(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 20)
        ibs = integrated_brier_score(model, X, time, event, eval_times)
        assert isinstance(ibs, float)

    def test_ibs_nonnegative(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 20)
        ibs = integrated_brier_score(model, X, time, event, eval_times)
        assert ibs >= 0

    def test_ibs_below_null(self, fitted_model_large):
        model, X, time, event = fitted_model_large
        eval_times = np.linspace(2, time.max() * 0.5, 30)
        ibs = integrated_brier_score(model, X, time, event, eval_times)
        assert ibs < 0.25

    def test_ibs_finite(self, fitted_model):
        model, X, time, event = fitted_model
        eval_times = np.linspace(1, time.max() * 0.7, 20)
        ibs = integrated_brier_score(model, X, time, event, eval_times)
        assert np.isfinite(ibs)

    def test_single_time_point_fallback(self, fitted_model):
        """With a single time point, falls back to mean (no division by zero)."""
        model, X, time, event = fitted_model
        ibs = integrated_brier_score(model, X, time, event,
                                     eval_times=np.array([10.0, 10.0]))
        assert np.isfinite(ibs)


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------

class TestConcordance:
    def test_returns_float(self, fitted_model):
        model, X, time, event = fitted_model
        c = concordance(model, X, time, event)
        assert isinstance(c, float)

    def test_range(self, fitted_model):
        model, X, time, event = fitted_model
        c = concordance(model, X, time, event)
        assert 0.0 <= c <= 1.0

    def test_above_chance_with_strong_signal(self, fitted_model_large):
        """With beta=1.0 and n=1000, C-index should be clearly above 0.5."""
        model, X, time, event = fitted_model_large
        c = concordance(model, X, time, event)
        assert c > 0.55

    def test_unfitted_raises(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            concordance(model, X, time, event)

    def test_finite(self, fitted_model):
        model, X, time, event = fitted_model
        c = concordance(model, X, time, event)
        assert np.isfinite(c)

    def test_cause_2(self, fitted_model):
        model, X, time, event = fitted_model
        c = concordance(model, X, time, event, cause=2)
        # Just ensure it runs and returns a float in range (or nan if no events)
        assert np.isnan(c) or (0.0 <= c <= 1.0)

    def test_no_events_returns_nan(self):
        """If no cause-1 events in the data, concordance should return nan."""
        X, time, event = simulate_competing_risks(n=200, beta=0.5, seed=5)
        # Fit on original data, test on subset with only cause-2 events
        model = FineGrayModel().fit(X, time, event)
        mask  = event == 2
        if mask.sum() < 5:
            pytest.skip("not enough cause-2 events for this test")
        c = concordance(model, X[mask], time[mask], event[mask], cause=1)
        assert np.isnan(c)
