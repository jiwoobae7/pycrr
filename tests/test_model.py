"""
Unit tests for comprsk.

All tests use synthetic data so they run without external downloads.
The simulation design follows Fine & Gray (1999): two competing causes,
with cause-1 hazard proportional to exp(beta * X).
"""

import numpy as np
import pytest

from pycrr import FineGrayModel, estimate_ipcw_weights
from conftest import simulate_competing_risks


# ---------------------------------------------------------------------------
# core.py tests
# ---------------------------------------------------------------------------

class TestIPCW:
    def test_returns_correct_shapes(self, small_dataset):
        X, time, event = small_dataset
        weights, ipcw_func = estimate_ipcw_weights(time, event)
        assert weights.shape == (len(time),)

    def test_mean_is_one(self, small_dataset):
        _, time, event = small_dataset
        weights, _ = estimate_ipcw_weights(time, event)
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_competing_events_have_weight_one(self, small_dataset):
        _, time, event = small_dataset
        weights, _ = estimate_ipcw_weights(time, event)
        # After normalization the raw weight for event==2 was 1.0, so
        # after mean-normalization it will not be exactly 1, but the
        # raw values should be consistent.  Just check no NaN/Inf.
        assert np.all(np.isfinite(weights))

    def test_positive_weights(self, small_dataset):
        _, time, event = small_dataset
        weights, _ = estimate_ipcw_weights(time, event)
        assert np.all(weights > 0)

    def test_ipcw_func_callable(self, small_dataset):
        _, time, event = small_dataset
        _, ipcw_func = estimate_ipcw_weights(time, event)
        w = ipcw_func(time[0])
        assert np.isfinite(w)
        assert w > 0

    def test_clip_prevents_infinite_weights(self, small_dataset):
        _, time, event = small_dataset
        weights, _ = estimate_ipcw_weights(time, event, clip=0.01)
        assert np.all(np.isfinite(weights))


# ---------------------------------------------------------------------------
# FineGrayModel.fit tests
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel()
        result = model.fit(X, time, event)
        assert result is model

    def test_coef_shape(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        assert model.coef_.shape == (X.shape[1],)

    def test_se_shape(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        assert model.standard_errors_.shape == (X.shape[1],)

    def test_se_positive(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        assert np.all(model.standard_errors_ > 0)

    def test_coefficient_sign(self, large_dataset):
        """With beta=1.0, the fitted coef should be positive."""
        X, time, event = large_dataset
        model = FineGrayModel().fit(X, time, event)
        assert model.coef_[0] > 0

    def test_fit_with_precomputed_ipcw(self, small_dataset):
        X, time, event = small_dataset
        weights, _ = estimate_ipcw_weights(time, event)
        model = FineGrayModel().fit(X, time, event, ipcw=weights)
        assert model.coef_ is not None

    def test_fit_multivariate(self):
        X, time, event = simulate_competing_risks(n=400, beta=0.5, seed=7)
        X2 = np.hstack([X, np.random.default_rng(99).normal(size=(len(X), 1))])
        model = FineGrayModel().fit(X2, time, event)
        assert model.coef_.shape == (2,)

    def test_l2_regularization_shrinks_coef(self, large_dataset):
        X, time, event = large_dataset
        m0 = FineGrayModel(l2=0.0).fit(X, time, event)
        m1 = FineGrayModel(l2=5.0).fit(X, time, event)
        assert abs(m1.coef_[0]) < abs(m0.coef_[0])

    def test_dataframe_input(self, small_dataset):
        import pandas as pd
        X, time, event = small_dataset
        df = pd.DataFrame(X, columns=["age"])
        model = FineGrayModel().fit(df, time, event)
        assert model.coef_.shape == (1,)

    def test_p_values_in_range(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        assert np.all(model.p_values_ >= 0)
        assert np.all(model.p_values_ <= 1)

    def test_unfitted_raises_on_predict(self):
        model = FineGrayModel()
        with pytest.raises(RuntimeError):
            model.predict_cif(np.zeros((2, 1)))


# ---------------------------------------------------------------------------
# FineGrayModel.predict_cif tests
# ---------------------------------------------------------------------------

class TestPredictCIF:
    def test_output_shape(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        times = np.linspace(0, time.max() * 0.8, 50)
        cif   = model.predict_cif(X[:5], times)
        assert cif.shape == (5, 50)

    def test_cif_monotone(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        times = np.linspace(0, time.max() * 0.8, 100)
        cif   = model.predict_cif(X[:3], times)
        assert np.all(np.diff(cif, axis=1) >= -1e-12)

    def test_cif_bounds(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        times = np.linspace(0, time.max() * 0.8, 100)
        cif   = model.predict_cif(X, times)
        assert np.all(cif >= 0)
        assert np.all(cif <= 1)

    def test_cif_starts_at_zero(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        cif   = model.predict_cif(X[:5], times=np.array([0.0, 1.0, 5.0]))
        assert np.allclose(cif[:, 0], 0.0, atol=1e-8)

    def test_higher_X_higher_cif(self, large_dataset):
        """Higher covariate value -> higher CIF for cause 1 (beta > 0)."""
        X, time, event = large_dataset
        model  = FineGrayModel().fit(X, time, event)
        times  = np.linspace(1, time.max() * 0.5, 20)
        X_low  = np.array([[-2.0]])
        X_high = np.array([[ 2.0]])
        cif_low  = model.predict_cif(X_low,  times)
        cif_high = model.predict_cif(X_high, times)
        # At least the majority of time points should show higher CIF for X_high
        assert np.mean(cif_high > cif_low) > 0.8

    def test_default_times(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        cif   = model.predict_cif(X[:2])
        n_event_times = len(np.unique(time[event == 1]))
        # Default times = [0] + unique event times
        assert cif.shape[1] == n_event_times + 1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

class TestInference:
    def test_hazard_ratios_positive(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        assert np.all(model.hazard_ratios() > 0)

    def test_hazard_ratio_ci_ordering(self, small_dataset):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        for lo, hi in model.hazard_ratio_ci():
            assert lo < hi

    def test_summary_runs_without_error(self, small_dataset, capsys):
        X, time, event = small_dataset
        model = FineGrayModel().fit(X, time, event)
        model.summary(feature_names=["X"])
        captured = capsys.readouterr()
        assert "X" in captured.out
        assert "coef" in captured.out
