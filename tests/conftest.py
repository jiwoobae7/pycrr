"""
Shared fixtures for comprsk test suite.
"""

import numpy as np
import pytest

from pycrr import FineGrayModel


def simulate_competing_risks(n=500, beta=0.5, seed=42):
    """
    Simulate a simple two-cause competing risks dataset.

    Cause-1 time ~ Exp(lambda_1 * exp(beta * X))
    Cause-2 time ~ Exp(lambda_2)
    Censoring    ~ Exp(lambda_c)
    """
    rng = np.random.default_rng(seed)
    X   = rng.normal(size=(n, 1))

    lam1 = 0.1 * np.exp(beta * X.ravel())
    lam2 = 0.05
    lamc = 0.05

    t1 = rng.exponential(1.0 / lam1)
    t2 = rng.exponential(1.0 / lam2, size=n)
    tc = rng.exponential(1.0 / lamc, size=n)

    time  = np.minimum.reduce([t1, t2, tc])
    event = np.where(
        (t1 < t2) & (t1 < tc), 1,
        np.where(t2 < tc, 2, 0)
    )
    return X, time, event


@pytest.fixture
def small_dataset():
    return simulate_competing_risks(n=300, beta=0.5, seed=0)


@pytest.fixture
def large_dataset():
    return simulate_competing_risks(n=1000, beta=1.0, seed=1)


@pytest.fixture
def fitted_model(small_dataset):
    X, time, event = small_dataset
    return FineGrayModel().fit(X, time, event), X, time, event


@pytest.fixture
def fitted_model_large(large_dataset):
    X, time, event = large_dataset
    return FineGrayModel().fit(X, time, event), X, time, event
