"""
Tests for comprsk.surv (Surv class) and its integration with fit() methods.
"""

import numpy as np
import pytest

from pycrr import AalenJohansen, FineGrayModel, Surv
from pycrr.compare import gray_test
from conftest import simulate_competing_risks


# ---------------------------------------------------------------------------
# Surv construction
# ---------------------------------------------------------------------------

class TestSurvConstruct:
    def test_basic(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert np.array_equal(sv.time,  time.astype(float))
        assert np.array_equal(sv.event, event.astype(int))

    def test_len(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert len(sv) == len(time)

    def test_unpack(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        t_out, e_out = sv
        assert np.array_equal(t_out, time.astype(float))
        assert np.array_equal(e_out, event.astype(int))

    def test_repr_contains_n(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert f"n={len(time)}" in repr(sv)

    def test_repr_contains_censored(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert "censored=" in repr(sv)

    def test_causes_property(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert set(sv.causes) == {1, 2}

    def test_n_events_property(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        assert sv.n_events == int(np.sum(event != 0))

    def test_event_table(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        tbl = sv.event_table()
        assert "censored" in tbl
        assert "cause_1" in tbl
        assert "cause_2" in tbl
        assert tbl["cause_1"] + tbl["cause_2"] + tbl["censored"] == len(time)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Surv(np.array([1.0, 2.0]), np.array([0, 1, 0]))

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            Surv(np.ones((3, 2)), np.zeros((3, 2)))

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Surv(np.array([-1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_list_inputs(self):
        sv = Surv([1.0, 2.0, 3.0], [0, 1, 2])
        assert len(sv) == 3


# ---------------------------------------------------------------------------
# Integration: FineGrayModel.fit(X, Surv(...))
# ---------------------------------------------------------------------------

class TestFineGraySurv:
    def test_fit_with_surv(self, small_dataset):
        X, time, event = small_dataset
        sv    = Surv(time, event)
        model = FineGrayModel().fit(X, sv)
        assert model.coef_ is not None
        assert model.coef_.shape == (1,)

    def test_surv_matches_explicit(self, small_dataset):
        X, time, event = small_dataset
        m_explicit = FineGrayModel().fit(X, time, event)
        m_surv     = FineGrayModel().fit(X, Surv(time, event))
        np.testing.assert_allclose(m_surv.coef_, m_explicit.coef_, rtol=1e-10)

    def test_surv_with_keyword_event_raises(self, small_dataset):
        X, time, event = small_dataset
        with pytest.raises(ValueError, match="not both"):
            FineGrayModel().fit(X, Surv(time, event), event)

    def test_missing_event_raises(self, small_dataset):
        X, time, event = small_dataset
        with pytest.raises((ValueError, TypeError)):
            FineGrayModel().fit(X, time)   # no event, not a Surv


# ---------------------------------------------------------------------------
# Integration: AalenJohansen.fit(Surv(...))
# ---------------------------------------------------------------------------

class TestAalenJohansenSurv:
    def test_fit_with_surv(self, small_dataset):
        _, time, event = small_dataset
        sv = Surv(time, event)
        aj = AalenJohansen().fit(sv)
        assert aj.causes_ == [1, 2]

    def test_surv_matches_explicit(self, small_dataset):
        _, time, event = small_dataset
        aj_explicit = AalenJohansen().fit(time, event)
        aj_surv     = AalenJohansen().fit(Surv(time, event))
        t1, c1 = aj_explicit.predict(1)
        t2, c2 = aj_surv.predict(1)
        np.testing.assert_allclose(c1, c2)

    def test_surv_with_extra_event_raises(self, small_dataset):
        _, time, event = small_dataset
        with pytest.raises(ValueError, match="not both"):
            AalenJohansen().fit(Surv(time, event), event)

    def test_missing_event_raises(self, small_dataset):
        _, time, event = small_dataset
        with pytest.raises((ValueError, TypeError)):
            AalenJohansen().fit(time)


# ---------------------------------------------------------------------------
# Integration: gray_test(Surv(...), group)
# ---------------------------------------------------------------------------

class TestGrayTestSurv:
    def test_gray_test_with_surv(self, small_dataset):
        _, time, event = small_dataset
        group  = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(Surv(time, event), group)
        assert result.df == 1
        assert 0.0 <= result.pvalue <= 1.0

    def test_surv_matches_explicit(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        r_explicit = gray_test(time, event, group)
        r_surv     = gray_test(Surv(time, event), group)
        assert r_surv.statistic == pytest.approx(r_explicit.statistic, rel=1e-10)
