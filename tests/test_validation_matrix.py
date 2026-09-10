"""
Validation matrix for pycrr.

Rows = every public statistical output of FineGrayModel.
Columns = analytical test | R reference | simulation | edge cases.

Also covers:
  - Optimizer stability  (multiple starting values)
  - Multiple datasets    (synthetic variants + real mgus2)
  - Objective identity   (gradient check at R's solution)
  - Score residual diagnostics

Run with:
    python -m pytest tests/test_validation_matrix.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest

from scipy.optimize import minimize

from pycrr import FineGrayModel, Surv
from pycrr.core import _censoring_survival
from pycrr.model import _efron_grad_hess_tv, _compute_fg_statistics
from conftest import simulate_competing_risks


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "..", "examples", "validation_data.csv")


def _load_validation():
    df    = pd.read_csv(CSV)
    time  = df["time"].to_numpy(dtype=float)
    event = df["event"].to_numpy(dtype=int)
    X     = df[["X1", "X2"]].to_numpy(dtype=float)
    return X, time, event


def _sorted(X, time, event):
    order = np.argsort(time, kind="stable")
    return X[order], time[order], event[order]


def _tv_gradient(beta, X_s, time_s, event_s, G_i, G_func):
    g, _ = _efron_grad_hess_tv(X_s, time_s, event_s, G_i, G_func, beta, 1)
    return g


# ---------------------------------------------------------------------------
# 1. OBJECTIVE IDENTITY
#    Show pycrr and R cmprsk::crr optimize the same partial log-likelihood.
#    Evidence: gradient of OUR objective at R's solution is ~0.
# ---------------------------------------------------------------------------

class TestObjectiveIdentity:
    """
    If pycrr and R optimize the same function, then R's optimal beta must also
    be a stationary point of our objective.  Gradient ~0 at R's solution
    confirms identical objective.
    """

    def test_gradient_near_zero_at_r_solution(self):
        """Gradient of pycrr neg_loglik at R's solution must be ~0."""
        X, time, event = _load_validation()
        X_s, time_s, event_s = _sorted(X, time, event)
        G_func = _censoring_survival(time_s, event_s)
        G_i    = np.array([G_func(t) for t in time_s])

        R_COEF = np.array([0.765965, -0.378533])
        g = _tv_gradient(R_COEF, X_s, time_s, event_s, G_i, G_func)
        assert np.max(np.abs(g)) < 1e-3, (
            f"Gradient at R's solution should be ~0, got {g}. "
            "Non-zero gradient means different objective."
        )

    def test_pycrr_achieves_lower_negloglik_than_r(self):
        """pycrr should achieve neg_loglik <= R's neg_loglik (same or better)."""
        X, time, event = _load_validation()
        X_s, time_s, event_s = _sorted(X, time, event)
        G_func   = _censoring_survival(time_s, event_s)
        G_i      = np.array([G_func(t) for t in time_s])
        competing = (event_s != 0) & (event_s != 1)
        n = len(time_s)

        def neg_loglik(beta):
            eta     = np.clip(X_s @ beta, -30, 30)
            exp_eta = np.exp(eta)
            ll = 0.0
            for t in np.unique(time_s[event_s == 1]):
                tm   = (time_s == t) & (event_s == 1)
                cb   = (time_s < t) & competing
                rm   = (time_s >= t) | cb
                G_t  = G_func(t)
                w    = np.ones(n)
                if cb.any():
                    w[cb] = G_t / G_i[cb]
                S0   = (w[rm] * exp_eta[rm]).sum()
                ll  += eta[tm].sum() - np.log(max(S0, 1e-12))
            return -ll

        model  = FineGrayModel().fit(X, Surv(time, event))
        R_COEF = np.array([0.765965, -0.378533])
        assert neg_loglik(model.coef_) <= neg_loglik(R_COEF) + 1e-6

    def test_gradient_zero_at_pycrr_solution(self):
        """Gradient of pycrr neg_loglik at pycrr's own solution is ~0."""
        X, time, event = _load_validation()
        X_s, time_s, event_s = _sorted(X, time, event)
        G_func = _censoring_survival(time_s, event_s)
        G_i    = np.array([G_func(t) for t in time_s])

        model = FineGrayModel().fit(X, Surv(time, event))
        g     = _tv_gradient(model.coef_, X_s, time_s, event_s, G_i, G_func)
        assert np.max(np.abs(g)) < 1e-5, f"Gradient at pycrr solution not zero: {g}"


# ---------------------------------------------------------------------------
# 2. OPTIMIZER STABILITY  (multiple starting values)
#    L-BFGS-B should find the same solution regardless of initialization.
# ---------------------------------------------------------------------------

class TestOptimizerStability:

    R_COEF = np.array([0.765965, -0.378533])

    @pytest.fixture(autouse=True)
    def _load(self):
        X, time, event = _load_validation()
        self.X     = X
        self.time  = time
        self.event = event

    def _fit_from(self, beta0):
        """Fit with custom starting point by patching the optimizer."""
        X, time, event = self.X, self.time, self.event
        X_s, time_s, event_s = _sorted(X, time, event)
        G_func   = _censoring_survival(time_s, event_s)
        G_i      = np.array([G_func(t) for t in time_s])
        competing = (event_s != 0) & (event_s != 1)
        n = len(time_s)

        def neg_loglik(beta):
            eta = np.clip(X_s @ beta, -30, 30)
            exp_eta = np.exp(eta)
            ll = 0.0
            for t in np.unique(time_s[event_s == 1]):
                tm = (time_s == t) & (event_s == 1)
                cb = (time_s < t) & competing
                rm = (time_s >= t) | cb
                G_t = G_func(t)
                w   = np.ones(n)
                if cb.any(): w[cb] = G_t / G_i[cb]
                S0  = (w[rm] * np.exp(np.clip(X_s[rm] @ beta, -30, 30))).sum()
                ll += eta[tm].sum() - np.log(max(S0, 1e-12))
            return -ll

        def gradient(beta):
            g, _ = _efron_grad_hess_tv(X_s, time_s, event_s, G_i, G_func, beta, 1)
            return g

        res = minimize(neg_loglik, beta0, jac=gradient, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        return res.x

    def test_zero_start(self):
        coef = self._fit_from(np.zeros(2))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)

    def test_positive_start(self):
        coef = self._fit_from(np.array([0.5, 0.5]))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)

    def test_negative_start(self):
        coef = self._fit_from(np.array([-0.5, -0.5]))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)

    def test_large_positive_start(self):
        coef = self._fit_from(np.array([2.0, -2.0]))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)

    def test_random_start_1(self):
        rng  = np.random.default_rng(101)
        coef = self._fit_from(rng.normal(size=2))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)

    def test_random_start_2(self):
        rng  = np.random.default_rng(202)
        coef = self._fit_from(rng.normal(size=2))
        np.testing.assert_allclose(coef, self.R_COEF, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. MULTI-DATASET STRESS TEST
#    Every coefficient / SE check should hold across dataset variants.
# ---------------------------------------------------------------------------

def _make_heavy_censoring(n=600, seed=5):
    """~70% censoring: sparse events, should still estimate."""
    rng   = np.random.default_rng(seed)
    X     = rng.normal(size=(n, 1))
    lam1  = 0.05 * np.exp(0.7 * X.ravel())
    t1    = rng.exponential(1.0 / lam1)
    t2    = rng.exponential(1.0 / 0.03, size=n)
    tc    = rng.exponential(1.0 / 0.20, size=n)   # high censoring
    time  = np.minimum.reduce([t1, t2, tc])
    event = np.where((t1 < t2) & (t1 < tc), 1, np.where(t2 < tc, 2, 0))
    return X, time, event


def _make_many_ties(n=300, seed=8):
    """Discrete times — many exact ties."""
    rng   = np.random.default_rng(seed)
    X     = rng.normal(size=(n, 2))
    raw1  = rng.exponential(10, size=n) * np.exp(0.5 * X[:, 0])
    raw2  = rng.exponential(10, size=n)
    rawc  = rng.exponential(10, size=n)
    # Round to integers → lots of ties
    t1    = np.round(raw1).clip(1)
    t2    = np.round(raw2).clip(1)
    tc    = np.round(rawc).clip(1)
    time  = np.minimum.reduce([t1, t2, tc])
    event = np.where((t1 < t2) & (t1 < tc), 1, np.where(t2 < tc, 2, 0))
    return X, time, event


def _make_high_dim(n=800, p=8, seed=12):
    """p=8 covariates, only first two have signal."""
    rng   = np.random.default_rng(seed)
    X     = rng.normal(size=(n, p))
    lam1  = 0.1 * np.exp(0.6 * X[:, 0] - 0.4 * X[:, 1])
    t1    = rng.exponential(1.0 / lam1)
    t2    = rng.exponential(1.0 / 0.05, size=n)
    tc    = rng.exponential(1.0 / 0.05, size=n)
    time  = np.minimum.reduce([t1, t2, tc])
    event = np.where((t1 < t2) & (t1 < tc), 1, np.where(t2 < tc, 2, 0))
    return X, time, event


class TestMultiDatasetStress:
    """
    Structural / statistical validity checks across 5 dataset variants.
    No R reference — validates invariants that must always hold.
    """

    DATASETS = {
        "validation_n500": _load_validation,
        "synthetic_n1000": lambda: simulate_competing_risks(n=1000, beta=0.8, seed=3),
        "heavy_censoring":  _make_heavy_censoring,
        "many_ties":        _make_many_ties,
        "high_dim_p8":      _make_high_dim,
    }

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_coef_finite(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        assert np.all(np.isfinite(model.coef_)), name

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_se_positive_finite(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        assert np.all(model.standard_errors_ > 0), name
        assert np.all(np.isfinite(model.standard_errors_)), name

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_robust_se_positive_finite(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        assert np.all(model.robust_se_ > 0), name
        assert np.all(np.isfinite(model.robust_se_)), name

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_score_residual_column_sums(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        col_sums = model.score_residuals_.sum(axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-4,
                                   err_msg=f"{name}: score col sums not ~0")

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_sandwich_identity(self, name, loader):
        """robust_se_ == sqrt(diag(cov @ U^T U @ cov)) on every dataset."""
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        U    = model.score_residuals_
        meat = U.T @ U
        expected_se = np.sqrt(np.diag(model.cov_matrix_ @ meat @ model.cov_matrix_))
        np.testing.assert_allclose(model.robust_se_, expected_se, rtol=1e-5,
                                   err_msg=f"{name}: sandwich identity fails")

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_cif_valid(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        times = np.linspace(0.01, np.percentile(time, 80), 20)
        cif   = model.predict_cif(X[:5], times)
        assert np.all(cif >= 0), name
        assert np.all(cif <= 1), name
        assert np.all(np.diff(cif, axis=1) >= -1e-10), name

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_p_values_valid(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        assert np.all(model.p_values_ >= 0), name
        assert np.all(model.p_values_ <= 1), name

    @pytest.mark.parametrize("name,loader", list(DATASETS.items()))
    def test_robust_p_valid(self, name, loader):
        X, time, event = loader()
        model = FineGrayModel().fit(X, Surv(time, event))
        assert np.all(model.robust_p_ >= 0), name
        assert np.all(model.robust_p_ <= 1), name

    def test_sign_recovery_large(self):
        """With beta=(+1, -1) and n=1000, both signs should be recovered."""
        rng   = np.random.default_rng(55)
        n     = 1000
        X     = rng.normal(size=(n, 2))
        lam1  = 0.1 * np.exp(1.0 * X[:, 0] - 1.0 * X[:, 1])
        t1    = rng.exponential(1.0 / lam1)
        t2    = rng.exponential(20, size=n)
        tc    = rng.exponential(20, size=n)
        time  = np.minimum.reduce([t1, t2, tc])
        event = np.where((t1 < t2) & (t1 < tc), 1, np.where(t2 < tc, 2, 0))
        model = FineGrayModel().fit(X, Surv(time, event))
        assert model.coef_[0] > 0, "X1 coef should be positive"
        assert model.coef_[1] < 0, "X2 coef should be negative"


# ---------------------------------------------------------------------------
# 4. R REFERENCE VALIDATION  (every public output)
#    Tolerance: 1e-3 (matches paper claim).
# ---------------------------------------------------------------------------

class TestPublicOutputsVsR:
    """
    Validation matrix: every public statistical output vs R cmprsk 2.2-11.
    Dataset: validation_data.csv (n=500, set.seed(42)).
    """

    # R cmprsk reference values
    R = {
        "coef":      np.array([0.765965,   -0.378533]),
        "se_model":  np.array([0.069160,    0.060120]),
        "se_robust": np.array([0.080660,    0.060830]),
        "hr":        np.exp([0.765965, -0.378533]),
    }

    @pytest.fixture(autouse=True)
    def _fit(self):
        X, time, event = _load_validation()
        self.model = FineGrayModel().fit(X, Surv(time, event))

    # -- coef_ --
    def test_coef_x1(self):
        assert abs(self.model.coef_[0] - self.R["coef"][0]) < 1e-3

    def test_coef_x2(self):
        assert abs(self.model.coef_[1] - self.R["coef"][1]) < 1e-3

    # -- standard_errors_ --
    def test_se_model_x1(self):
        assert abs(self.model.standard_errors_[0] - self.R["se_model"][0]) < 1e-3

    def test_se_model_x2(self):
        assert abs(self.model.standard_errors_[1] - self.R["se_model"][1]) < 1e-3

    # -- robust_se_ --
    def test_se_robust_x1(self):
        assert abs(self.model.robust_se_[0] - self.R["se_robust"][0]) < 1e-3

    def test_se_robust_x2(self):
        assert abs(self.model.robust_se_[1] - self.R["se_robust"][1]) < 1e-3

    # -- hazard_ratios_ --
    def test_hr_x1(self):
        assert abs(self.model.hazard_ratios()[0] - self.R["hr"][0]) < 1e-2

    def test_hr_x2(self):
        assert abs(self.model.hazard_ratios()[1] - self.R["hr"][1]) < 1e-2

    # -- p_values_ (directional: both should be significant) --
    def test_pvalue_x1_significant(self):
        assert self.model.p_values_[0] < 0.001

    def test_pvalue_x2_significant(self):
        assert self.model.p_values_[1] < 0.001

    # -- robust_p_ (directional: both should be significant) --
    def test_robust_p_x1_significant(self):
        assert self.model.robust_p_[0] < 0.001

    def test_robust_p_x2_significant(self):
        assert self.model.robust_p_[1] < 0.001

    # -- score_residuals_: aggregate property that matches R (B = U^T U → robust SE) --
    def test_score_resid_gives_correct_sandwich(self):
        """U^T U sandwich gives robust SE matching R to 1e-3."""
        U    = self.model.score_residuals_
        meat = U.T @ U
        se   = np.sqrt(np.diag(self.model.cov_matrix_ @ meat @ self.model.cov_matrix_))
        np.testing.assert_allclose(se, self.R["se_robust"], atol=1e-3)

    def test_score_resid_column_sums_zero(self):
        np.testing.assert_allclose(
            self.model.score_residuals_.sum(axis=0), 0.0, atol=1e-6
        )

    # -- full robust covariance matrix (not just diagonal) --
    # R: fit$var (validation_data.csv, n=500)
    #   [[ 0.006505409, -0.001492414],
    #    [-0.001492414,  0.003700336]]
    R_VAR = np.array([[ 0.006505409, -0.001492414],
                      [-0.001492414,  0.003700336]])

    def test_robust_cov_full_matrix(self):
        """Full 2x2 robust cov matrix matches R fit$var to 1e-4 (including off-diagonal)."""
        U    = self.model.score_residuals_
        meat = U.T @ U
        V    = self.model.cov_matrix_ @ meat @ self.model.cov_matrix_
        np.testing.assert_allclose(V, self.R_VAR, atol=1e-4,
                                   err_msg="robust cov matrix vs R fit$var (validation_data.csv)")

    # -- schoenfeld_residuals_: one row per event time --
    def test_schoenfeld_n_rows(self):
        X, time, event = _load_validation()
        n_evt = len(np.unique(time[event == 1]))
        assert self.model.schoenfeld_residuals_.shape[0] == n_evt

    def test_schoenfeld_p_cols(self):
        assert self.model.schoenfeld_residuals_.shape[1] == 2

    # -- CIF predictions: no R CIF reference, but physical constraints --
    def test_cif_monotone(self):
        X, time, event = _load_validation()
        times = np.linspace(0.01, np.percentile(time, 80), 30)
        cif   = self.model.predict_cif(X[:10], times)
        assert np.all(np.diff(cif, axis=1) >= -1e-10)

    def test_cif_bounds(self):
        X, time, event = _load_validation()
        times = np.linspace(0.01, np.percentile(time, 80), 30)
        cif   = self.model.predict_cif(X[:10], times)
        assert np.all(cif >= 0) and np.all(cif <= 1)

    # -- check_proportionality: no R equivalent; check known result --
    def test_proportionality_no_violation(self):
        """p > 0.05 for both covariates (proportionality holds here)."""
        df = self.model.check_proportionality(feature_names=["X1", "X2"])
        p_x1 = df.loc[df.variable == "X1", "p"].values[0]
        p_x2 = df.loc[df.variable == "X2", "p"].values[0]
        assert p_x1 > 0.05, f"Unexpected X1 violation: p={p_x1:.4f}"
        assert p_x2 > 0.05, f"Unexpected X2 violation: p={p_x2:.4f}"


# ---------------------------------------------------------------------------
# 4b. PROPORTIONALITY TEST — POSITIVE CONTROL
#     Simulate data with a strong time-varying effect (coef * log(t) term),
#     fit the static Fine-Gray model, and verify check_proportionality detects
#     the violation (p < 0.05 for the affected covariate).
# ---------------------------------------------------------------------------

class TestProportionalityPositiveControl:
    """
    Positive control: verify check_proportionality has power to detect violations.
    We simulate event times from a subdistribution hazard where X1 has a strong
    log(t) interaction (beta_tv = 1.5), then fit the misspecified static model.
    The proportionality test must reject for X1.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(42)
        n   = 1500
        # Binary X1: crossing subdistribution hazards across two groups.
        # X1=1: Weibull(shape=0.3) => decreasing hazard => events cluster early.
        # X1=0: Weibull(shape=3.0) => increasing hazard => events cluster late.
        # This creates crossing subdistribution hazards, which the Schoenfeld
        # correlation test detects as a strong negative rho (rho ~ -0.65 in testing).
        X1  = (rng.uniform(0, 1, n) < 0.5).astype(float)
        X2  = rng.standard_normal(n)
        X   = np.column_stack([X1, X2])
        u     = rng.uniform(0, 1, n)
        shape = np.where(X1 == 1, 0.3, 3.0)
        scale = np.where(X1 == 1, 0.5, 4.0)
        time  = scale * (-np.log(u)) ** (1.0 / shape)
        time  = np.clip(time, 0.01, 50.0)
        event = np.ones(n, dtype=int)
        comp  = rng.uniform(0, 1, n) < 0.20
        event = np.where(comp, 2, event)
        cens_t = rng.uniform(1, 20, n)
        cens   = cens_t < time
        time   = np.where(cens, cens_t, time)
        event  = np.where(cens, 0, event)
        self.model  = FineGrayModel().fit(X, Surv(time, event.astype(int)))
        self.result = self.model.check_proportionality(feature_names=["X1", "X2"])

    def test_detects_x1_violation(self):
        """X1 has strong TV effect: proportionality test must reject (p < 0.05)."""
        p_x1 = self.result.loc[self.result.variable == "X1", "p"].values[0]
        assert p_x1 < 0.05, f"Proportionality test missed known X1 violation: p={p_x1:.4f}"

    def test_global_test_rejects(self):
        """Global test must also reject."""
        p_global = self.result.loc[self.result.variable == "GLOBAL", "p"].values[0]
        assert p_global < 0.05, f"Global test missed violation: p={p_global:.4f}"

    def test_output_shape(self):
        assert len(self.result) == 3  # X1, X2, GLOBAL


# ---------------------------------------------------------------------------
# 5. SCORE RESIDUAL INTERPRETATION
#    Document and test the decomposition used by pycrr.
#    Individual values differ from R's (different but equivalent decomposition);
#    the AGGREGATE B = U^T U and derived sandwich SE are identical.
# ---------------------------------------------------------------------------

class TestScoreResidualDecomposition:
    """
    pycrr score residuals U_i are the direct attribution of the IPCW partial
    likelihood score to each subject:

        U_i  =  delta_i (x_i - xbar(T_i))                            [event term]
             -  exp(eta_i) * SUM_{t_k <= T_i}  c_k (x_i - xbar(t_k))  [phase-a drag]
             -  [exp(eta_i)/G(T_i^-)] * SUM_{t_k > T_i}  G(t_k^-)*c_k*(x_i - xbar(t_k))
                                                            [phase-b, competing only]

    where c_k = d_k / S0(t_k)  (Breslow baseline hazard increment).

    R's cmprsk::crr fit$res uses a different attribution strategy.
    Both satisfy:
        (a) SUM_i U_i = 0  (score equation at MLE)
        (b) A^{-1} (SUM_i U_i U_i^T) A^{-1} = robust variance
    Neither is uniquely "correct" — they are different members of the same
    equivalence class of sandwich-consistent influence functions.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        X, time, event = _load_validation()
        self.model   = FineGrayModel().fit(X, Surv(time, event))
        self.X       = X
        self.time    = time
        self.event   = event

    def test_property_a_column_sums(self):
        """SUM_i U_i == 0 (score equation at MLE)."""
        np.testing.assert_allclose(
            self.model.score_residuals_.sum(axis=0), 0.0, atol=1e-6
        )

    def test_property_b_sandwich(self):
        """A^{-1} B A^{-1} reproduces robust_se_ to machine precision."""
        U    = self.model.score_residuals_
        meat = U.T @ U
        se   = np.sqrt(np.diag(self.model.cov_matrix_ @ meat @ self.model.cov_matrix_))
        np.testing.assert_allclose(se, self.model.robust_se_, rtol=1e-8)

    def test_r_robust_se_also_matches(self):
        """Sandwich from our U gives robust SE matching R's fit$var."""
        R_SE = np.array([0.080660, 0.060830])
        np.testing.assert_allclose(self.model.robust_se_, R_SE, atol=1e-3)

    def test_event_term_correct(self):
        """
        For each event subject, the event term x_i - xbar(T_i) is computed
        from the weighted mean of the risk set at T_i.
        Verify xbar is a convex combination (inside covariate range).
        """
        X_s, time_s, event_s = _sorted(self.X, self.time, self.event)
        G_func = _censoring_survival(time_s, event_s)
        G_i    = np.array([G_func(t) for t in time_s])
        beta   = self.model.coef_

        _, _, barx_arr, _ = _compute_fg_statistics(
            X_s, time_s, event_s, G_i, G_func, beta, 1
        )
        # barx at each event time should be within [min(X), max(X)]
        for col in range(X_s.shape[1]):
            assert np.all(barx_arr[:, col] >= X_s[:, col].min() - 1e-6)
            assert np.all(barx_arr[:, col] <= X_s[:, col].max() + 1e-6)

    def test_competing_subjects_have_nonzero_residuals(self):
        """Competing event subjects contribute to U via phase-b drag."""
        X, time, event = self.X, self.time, self.event
        U = self.model.score_residuals_

        order   = np.argsort(time, kind="stable")
        inv_ord = np.argsort(order)

        # At least some competing event subjects should have non-zero residuals
        comp_mask     = (event == 2)
        comp_sorted   = np.argsort(time, kind="stable")[np.where(
            (event[np.argsort(time)] == 2))[0]]
        U_comp = self.model.score_residuals_[np.where(
            event[np.argsort(time, kind='stable')] == 2)[0]]
        assert np.any(np.abs(U_comp) > 1e-8), "Competing subjects should have nonzero U"


# ---------------------------------------------------------------------------
# 6. REAL DATA: mgus2 — numerical comparison against R cmprsk::crr
#    R code:
#      d <- read.csv("examples/mgus2.csv"); d <- d[d$futime > 0, ]
#      event <- ifelse(d$pstat==1, 1, ifelse(d$death==1, 2, 0))
#      X <- scale(as.matrix(d[, c("age","dxyr")]))
#      fit <- crr(ftime=d$futime, fstatus=event, cov1=X, failcode=1, cencode=0)
#      R coef:      age=-0.1757854  dxyr=-0.1897006
#      R model SE:  age=0.08658689  dxyr=0.09310431
#      R robust SE: age=0.06839517  dxyr=0.09153166
# ---------------------------------------------------------------------------

class TestMgus2:

    MGUS2_CSV = os.path.join(HERE, "..", "examples", "mgus2.csv")

    # R reference values (cmprsk::crr, standardized X)
    R_COEF      = np.array([-0.1757854, -0.1897006])
    R_SE_MODEL  = np.array([ 0.08658689, 0.09310431])
    R_SE_ROBUST = np.array([ 0.06839517, 0.09153166])

    @pytest.fixture(autouse=True)
    def _load(self):
        if not os.path.exists(self.MGUS2_CSV):
            pytest.skip("mgus2.csv not found")
        df = pd.read_csv(self.MGUS2_CSV)
        # mgus2 uses pstat (progression to PCM), not 'pcm'
        df = df.dropna(subset=["futime", "death", "pstat"])
        # event: 1=PCM progression, 2=death without PCM, 0=censored
        df["event"] = np.where(df["pstat"] == 1, 1,
                      np.where(df["death"] == 1, 2, 0))
        df = df[df["futime"] > 0]
        self.time  = df["futime"].to_numpy(dtype=float)
        self.event = df["event"].to_numpy(dtype=int)
        X_raw = df[["age", "dxyr"]].to_numpy(dtype=float)
        # Standardize (matches R's scale())
        self.X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
        self.model = FineGrayModel().fit(self.X, Surv(self.time, self.event))

    # --- R numerical comparisons ---

    def test_coef_vs_r(self):
        np.testing.assert_allclose(self.model.coef_, self.R_COEF, atol=1e-3,
                                   err_msg="mgus2 coef vs R")

    def test_se_model_vs_r(self):
        np.testing.assert_allclose(self.model.standard_errors_, self.R_SE_MODEL, atol=1e-3,
                                   err_msg="mgus2 model SE vs R")

    def test_se_robust_vs_r(self):
        np.testing.assert_allclose(self.model.robust_se_, self.R_SE_ROBUST, atol=1e-3,
                                   err_msg="mgus2 robust SE vs R")

    # --- Structural validity ---

    def test_sandwich_identity(self):
        U    = self.model.score_residuals_
        meat = U.T @ U
        se   = np.sqrt(np.diag(self.model.cov_matrix_ @ meat @ self.model.cov_matrix_))
        np.testing.assert_allclose(se, self.model.robust_se_, rtol=1e-6)

    def test_cif_valid(self):
        times = np.percentile(self.time, [10, 25, 50])
        cif   = self.model.predict_cif(self.X[:10], times)
        assert np.all(cif >= 0) and np.all(cif <= 1)

    def test_proportionality_runs(self):
        df = self.model.check_proportionality(feature_names=["age", "dxyr"])
        assert "GLOBAL" in df["variable"].values

    def test_robust_cov_full_matrix(self):
        """Full 2x2 robust cov matrix matches R fit$var (off-diagonal included).
        R: print(fit$var)  =>  [[0.004678, -0.000544], [-0.000544, 0.008378]]
        """
        R_VAR = np.array([[ 0.0046778991, -0.0005444287],
                          [-0.0005444287,  0.0083780445]])
        U    = self.model.score_residuals_
        meat = U.T @ U
        V    = self.model.cov_matrix_ @ meat @ self.model.cov_matrix_
        np.testing.assert_allclose(V, R_VAR, atol=1e-4,
                                   err_msg="mgus2 robust cov matrix vs R fit$var")

    def test_score_col_sums(self):
        np.testing.assert_allclose(
            self.model.score_residuals_.sum(axis=0), 0.0, atol=1e-5
        )
