"""
Tests for comprsk.compare (gray_test, GrayTestResult).
"""

import numpy as np
import pytest

from pycrr.compare import GrayTestResult, gray_test
from conftest import simulate_competing_risks


def _two_group_data(n=400, beta_diff=1.5, seed=0):
    """Two groups with clearly different CIFs."""
    rng = np.random.default_rng(seed)

    # Group 0: low hazard
    t1_0 = rng.exponential(20, size=n // 2)
    t2_0 = rng.exponential(10, size=n // 2)
    tc_0 = rng.exponential(15, size=n // 2)

    # Group 1: high hazard for cause 1
    t1_1 = rng.exponential(5, size=n // 2)
    t2_1 = rng.exponential(10, size=n // 2)
    tc_1 = rng.exponential(15, size=n // 2)

    time0  = np.minimum.reduce([t1_0, t2_0, tc_0])
    event0 = np.where((t1_0 < t2_0) & (t1_0 < tc_0), 1,
                      np.where(t2_0 < tc_0, 2, 0))

    time1  = np.minimum.reduce([t1_1, t2_1, tc_1])
    event1 = np.where((t1_1 < t2_1) & (t1_1 < tc_1), 1,
                      np.where(t2_1 < tc_1, 2, 0))

    time  = np.concatenate([time0,  time1])
    event = np.concatenate([event0, event1])
    group = np.array([0] * (n // 2) + [1] * (n // 2))
    return time, event, group


def _same_group_data(n=400, seed=99):
    """Two groups drawn from identical distributions."""
    rng   = np.random.default_rng(seed)
    time  = rng.exponential(10, size=n)
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])
    group = rng.choice([0, 1], size=n)
    return time, event, group


class TestGrayTestResult:
    def test_repr(self):
        result = GrayTestResult(statistic=3.14, pvalue=0.08,
                                df=1, group_stats={0: 1.0, 1: -1.0},
                                groups=[0, 1])
        r = repr(result)
        assert "3.1400" in r
        assert "pvalue" in r

    def test_summary_prints(self, capsys):
        result = GrayTestResult(statistic=5.0, pvalue=0.025,
                                df=1, group_stats={0: 2.0, 1: -2.0},
                                groups=[0, 1])
        result.summary()
        out = capsys.readouterr().out
        assert "Gray" in out
        assert "5.0" in out or "5.00" in out


class TestGrayTest:
    def test_returns_gray_test_result(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group)
        assert isinstance(result, GrayTestResult)

    def test_two_groups_df_one(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group)
        assert result.df == 1

    def test_three_groups_df_two(self, small_dataset):
        _, time, event = small_dataset
        n = len(time)
        group = np.zeros(n, dtype=int)
        group[n // 3: 2 * n // 3] = 1
        group[2 * n // 3:]        = 2
        result = gray_test(time, event, group)
        assert result.df == 2

    def test_statistic_nonnegative(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group)
        assert result.statistic >= 0

    def test_pvalue_in_unit_interval(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group)
        assert 0.0 <= result.pvalue <= 1.0

    def test_different_groups_low_pvalue(self):
        time, event, group = _two_group_data(n=600, seed=42)
        result = gray_test(time, event, group)
        # Groups differ substantially; expect p < 0.05 with n=600
        assert result.pvalue < 0.05

    def test_same_distribution_high_pvalue(self):
        time, event, group = _same_group_data(n=600, seed=7)
        result = gray_test(time, event, group)
        # Groups are identical; p should not be very small
        # (not a guaranteed test, but should rarely fail under H0)
        assert result.pvalue > 1e-4

    def test_group_stats_keys(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, "A", "B")
        result = gray_test(time, event, group)
        assert set(result.group_stats.keys()) == {"A", "B"}

    def test_group_stats_sum_near_zero(self, small_dataset):
        """U statistics sum to 0 by construction."""
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group)
        total_U = sum(result.group_stats.values())
        assert abs(total_U) < 1e-8

    def test_fewer_than_two_groups_raises(self, small_dataset):
        _, time, event = small_dataset
        group = np.zeros(len(time), dtype=int)
        with pytest.raises(ValueError, match="at least 2 groups"):
            gray_test(time, event, group)

    def test_no_cause_events_raises(self):
        time  = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([0, 2, 2, 0])
        group = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="No events of cause 1"):
            gray_test(time, event, group, cause=1)

    def test_rho_parameter_runs(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group, rho=1)
        assert isinstance(result, GrayTestResult)
        assert np.isfinite(result.statistic)

    def test_cause_two(self, small_dataset):
        _, time, event = small_dataset
        group = np.where(np.arange(len(time)) < len(time) // 2, 0, 1)
        result = gray_test(time, event, group, cause=2)
        assert isinstance(result, GrayTestResult)
        assert result.df == 1
