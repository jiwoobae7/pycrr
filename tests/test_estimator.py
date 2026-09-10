"""
Tests for comprsk.estimator (AalenJohansen).
"""

import numpy as np
import pytest

from pycrr.estimator import AalenJohansen
from conftest import simulate_competing_risks


class TestAalenJohansenFit:
    def test_fit_returns_self(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen()
        assert aj.fit(time, event) is aj

    def test_causes_detected(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        assert set(aj.causes_) == {1, 2}

    def test_n_set(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        assert aj.n_ == len(time)

    def test_no_events_raises(self):
        time  = np.array([1.0, 2.0, 3.0])
        event = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="No events"):
            AalenJohansen().fit(time, event)

    def test_single_cause(self):
        rng   = np.random.default_rng(7)
        time  = rng.exponential(5, size=200)
        event = rng.choice([0, 1], size=200)
        aj    = AalenJohansen().fit(time, event)
        assert aj.causes_ == [1]

    def test_tables_populated(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        for cause in [1, 2]:
            tbl = aj._tables[cause]
            assert "times" in tbl and "cif" in tbl and "var" in tbl


class TestAalenJohansenPredict:
    def test_predict_default_times_shape(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        t, cif = aj.predict(1)
        assert t.shape == cif.shape
        assert t[0] == 0.0

    def test_predict_custom_times(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        grid = np.linspace(0, time.max() * 0.9, 30)
        t_out, cif = aj.predict(1, times=grid)
        assert cif.shape == (30,)
        assert np.allclose(t_out, grid)

    def test_cif_starts_at_zero(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        _, cif = aj.predict(1)
        assert cif[0] == pytest.approx(0.0)

    def test_cif_nondecreasing(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        _, cif = aj.predict(1)
        assert np.all(np.diff(cif) >= -1e-12)

    def test_cif_in_unit_interval(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        for cause in [1, 2]:
            _, cif = aj.predict(cause)
            assert np.all(cif >= 0)
            assert np.all(cif <= 1)

    def test_sum_of_cifs_le_one(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        grid = np.linspace(0, time.max() * 0.8, 50)
        _, cif1 = aj.predict(1, times=grid)
        _, cif2 = aj.predict(2, times=grid)
        assert np.all(cif1 + cif2 <= 1.0 + 1e-10)

    def test_predict_extrapolation_clamps(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        t_way_out = np.array([time.max() * 100])
        _, cif = aj.predict(1, times=t_way_out)
        assert np.all(np.isfinite(cif))

    def test_unfitted_raises(self):
        aj = AalenJohansen()
        with pytest.raises(RuntimeError, match="not fitted"):
            aj.predict(1)

    def test_unknown_cause_raises(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        with pytest.raises(ValueError, match="Cause 9"):
            aj.predict(9)


class TestAalenJohansenCI:
    def test_ci_shape(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        lo, hi = aj.confidence_intervals(1)
        _, cif = aj.predict(1)
        assert lo.shape == cif.shape
        assert hi.shape == cif.shape

    def test_ci_ordering(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        # Skip t=0 where CIF=0 gives degenerate CI
        grid = np.linspace(0.5, 30.0, 40)
        lo, hi = aj.confidence_intervals(1, times=grid)
        _, cif = aj.predict(1, times=grid)
        # Where CIF > 0, lo <= cif <= hi
        mask = cif > 0
        assert np.all(lo[mask] <= cif[mask] + 1e-10)
        assert np.all(hi[mask] >= cif[mask] - 1e-10)

    def test_ci_bounds_in_unit_interval(self, small_dataset):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        lo, hi = aj.confidence_intervals(1)
        # NaN is expected at t=0 where CIF=0 (log-log transform is degenerate there)
        finite = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[finite] >= 0)
        assert np.all(hi[finite] <= 1)

    def test_wider_ci_at_90pct(self, small_dataset):
        """90% CI should be narrower than 95% CI."""
        _, time, event = small_dataset
        aj95 = AalenJohansen(confidence_level=0.95).fit(time, event)
        aj90 = AalenJohansen(confidence_level=0.90).fit(time, event)
        grid = np.linspace(1, 30, 20)
        lo95, hi95 = aj95.confidence_intervals(1, times=grid)
        lo90, hi90 = aj90.confidence_intervals(1, times=grid)
        width95 = hi95 - lo95
        width90 = hi90 - lo90
        assert np.mean(width95 >= width90) > 0.8


class TestAalenJohansenSummary:
    def test_summary_prints(self, small_dataset, capsys):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        aj.summary(cause=1)
        out = capsys.readouterr().out
        assert "CIF" in out
        assert "Time" in out

    def test_summary_custom_times(self, small_dataset, capsys):
        _, time, event = small_dataset
        aj = AalenJohansen().fit(time, event)
        aj.summary(cause=1, times=[5.0, 10.0, 20.0])
        out = capsys.readouterr().out
        assert "5.0" in out or "5." in out
