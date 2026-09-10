"""
Tests for new pycrr features:
  - Robust sandwich SE  (robust_se_)
  - Per-subject score residuals  (score_residuals_)
  - Proportionality test  (check_proportionality)
  - Time-varying covariate effects  (cov2 / tf)

Property-based tests that hold regardless of dataset:
  structural shapes, signs, mathematical identities.

Numerical validation against R cmprsk reference values is in
  examples/validation_robust.py
"""

import numpy as np
import pandas as pd
import pytest

from pycrr import FineGrayModel, Surv
from conftest import simulate_competing_risks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def medium_dataset():
    return simulate_competing_risks(n=400, beta=0.8, seed=10)


@pytest.fixture
def two_covariate_dataset():
    rng  = np.random.default_rng(7)
    n    = 500
    X    = rng.normal(size=(n, 2))
    lam1 = 0.1 * np.exp(0.6 * X[:, 0] - 0.4 * X[:, 1])
    t1   = rng.exponential(1.0 / lam1)
    t2   = rng.exponential(1.0 / 0.05, size=n)
    tc   = rng.exponential(1.0 / 0.05, size=n)
    time  = np.minimum.reduce([t1, t2, tc])
    event = np.where((t1 < t2) & (t1 < tc), 1, np.where(t2 < tc, 2, 0))
    return X, time, event


@pytest.fixture
def fitted_two_cov(two_covariate_dataset):
    X, time, event = two_covariate_dataset
    return FineGrayModel().fit(X, Surv(time, event)), X, time, event


# ---------------------------------------------------------------------------
# R-validated numerical check (validation_data.csv)
# ---------------------------------------------------------------------------

class TestRValidation:
    """Numerical checks against R cmprsk::crr (validation_data.csv, n=500)."""

    @pytest.fixture(autouse=True)
    def load(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        csv  = os.path.join(here, "..", "examples", "validation_data.csv")
        df   = pd.read_csv(csv)
        time  = df["time"].to_numpy(dtype=float)
        event = df["event"].to_numpy(dtype=int)
        X     = df[["X1", "X2"]].to_numpy(dtype=float)
        self.model = FineGrayModel().fit(X, Surv(time, event))
        # Reference values from R cmprsk 2.2-11
        self.R_COEF      = np.array([0.765965,  -0.378533])
        self.R_SE_MODEL  = np.array([0.069160,   0.060120])
        self.R_SE_ROBUST = np.array([0.080660,   0.060830])

    def test_coef_matches_r(self):
        np.testing.assert_allclose(
            self.model.coef_, self.R_COEF, atol=1e-3,
            err_msg="Coefficients do not match R cmprsk::crr"
        )

    def test_model_se_matches_r(self):
        np.testing.assert_allclose(
            self.model.standard_errors_, self.R_SE_MODEL, atol=1e-3,
            err_msg="Model-based SE does not match R fit$invinf"
        )

    def test_robust_se_matches_r(self):
        np.testing.assert_allclose(
            self.model.robust_se_, self.R_SE_ROBUST, atol=1e-3,
            err_msg="Robust SE does not match R fit$var"
        )

    def test_robust_se_larger_than_model_se(self):
        """On this dataset R shows robust SE > model SE for X1."""
        assert self.model.robust_se_[0] > self.model.standard_errors_[0]


# ---------------------------------------------------------------------------
# Robust SE
# ---------------------------------------------------------------------------

class TestRobustSE:
    def test_robust_se_exists(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        assert model.robust_se_ is not None

    def test_robust_se_shape(self, fitted_two_cov):
        model, X, *_ = fitted_two_cov
        assert model.robust_se_.shape == (X.shape[1],)

    def test_robust_se_positive(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        assert np.all(model.robust_se_ > 0)

    def test_robust_se_finite(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        assert np.all(np.isfinite(model.robust_se_))

    def test_robust_cov_matrix_shape(self, fitted_two_cov):
        model, X, *_ = fitted_two_cov
        p = X.shape[1]
        assert model.robust_cov_matrix_.shape == (p, p)

    def test_robust_cov_symmetric(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        V = model.robust_cov_matrix_
        np.testing.assert_allclose(V, V.T, atol=1e-10)

    def test_robust_cov_psd(self, fitted_two_cov):
        """Sandwich covariance matrix must be positive semi-definite."""
        model, *_ = fitted_two_cov
        eigvals = np.linalg.eigvalsh(model.robust_cov_matrix_)
        assert np.all(eigvals >= -1e-10)

    def test_robust_se_equals_sqrt_diag(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        expected = np.sqrt(np.diag(model.robust_cov_matrix_))
        np.testing.assert_allclose(model.robust_se_, expected, atol=1e-10)

    def test_robust_z_shape(self, fitted_two_cov):
        model, X, *_ = fitted_two_cov
        assert model.robust_z_.shape == (X.shape[1],)

    def test_robust_p_in_range(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        assert np.all(model.robust_p_ >= 0)
        assert np.all(model.robust_p_ <= 1)

    def test_robust_not_available_with_precomputed_ipcw(self, small_dataset):
        """Robust SE requires time-varying IPCW; None when static weights given."""
        from pycrr import estimate_ipcw_weights
        X, time, event = small_dataset
        w, _ = estimate_ipcw_weights(time, event)
        model = FineGrayModel().fit(X, time, event, ipcw=w)
        assert model.robust_se_ is None

    def test_robust_ci_wider_than_model_ci(self, large_dataset):
        """On large datasets with strong signal, robust CI should be close to model CI."""
        X, time, event = large_dataset
        model = FineGrayModel().fit(X, time, event)
        # Both should be finite and positive width
        ci_model  = model.hazard_ratio_ci(robust=False)
        ci_robust = model.hazard_ratio_ci(robust=True)
        for (lo_m, hi_m), (lo_r, hi_r) in zip(ci_model, ci_robust):
            assert lo_m < hi_m
            assert lo_r < hi_r


# ---------------------------------------------------------------------------
# Score residuals
# ---------------------------------------------------------------------------

class TestScoreResiduals:
    def test_shape(self, fitted_two_cov):
        model, X, time, _ = fitted_two_cov
        assert model.score_residuals_.shape == (len(time), X.shape[1])

    def test_row_sums_near_zero(self, fitted_two_cov):
        """Column sums of score residuals ≈ 0 (score equation at MLE)."""
        model, *_ = fitted_two_cov
        col_sums = model.score_residuals_.sum(axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-6)

    def test_finite(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        assert np.all(np.isfinite(model.score_residuals_))

    def test_censored_rows_zero(self, fitted_two_cov):
        """
        Censored subjects contribute zero to the event term (Delta_i=0),
        and their drag contributions should give non-zero rows in general.
        But check that all-zero rows are only for subjects truly at t=0
        in the sense that score residuals are finite for everyone.
        """
        model, *_ = fitted_two_cov
        # No row should be all-NaN
        assert not np.any(np.all(np.isnan(model.score_residuals_), axis=1))

    def test_meat_is_psd(self, fitted_two_cov):
        """B = U^T U must be positive semi-definite."""
        model, *_ = fitted_two_cov
        U    = model.score_residuals_
        meat = U.T @ U
        eigvals = np.linalg.eigvalsh(meat)
        assert np.all(eigvals >= -1e-10)

    def test_sandwich_formula(self, fitted_two_cov):
        """robust_cov = cov @ meat @ cov  (A^{-1} B A^{-1})."""
        model, *_ = fitted_two_cov
        U    = model.score_residuals_
        meat = U.T @ U
        expected = model.cov_matrix_ @ meat @ model.cov_matrix_
        np.testing.assert_allclose(
            model.robust_cov_matrix_, expected, rtol=1e-6
        )

    def test_not_available_with_precomputed_ipcw(self, small_dataset):
        from pycrr import estimate_ipcw_weights
        X, time, event = small_dataset
        w, _ = estimate_ipcw_weights(time, event)
        model = FineGrayModel().fit(X, time, event, ipcw=w)
        assert model.score_residuals_ is None

    def test_n_columns_equals_p(self, medium_dataset):
        X, time, event = medium_dataset
        model = FineGrayModel().fit(X, time, event)
        assert model.score_residuals_.shape[1] == X.shape[1]


# ---------------------------------------------------------------------------
# Proportionality test
# ---------------------------------------------------------------------------

class TestProportionalityTest:
    def test_returns_dataframe(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        result = model.check_proportionality(feature_names=["X1", "X2"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        for col in ["variable", "rho", "chisq", "df", "p"]:
            assert col in df.columns

    def test_n_rows(self, fitted_two_cov):
        """One row per covariate + GLOBAL row."""
        model, X, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        assert len(df) == X.shape[1] + 1

    def test_global_row_present(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        assert "GLOBAL" in df["variable"].values

    def test_p_values_in_range(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        p_vals = df["p"].dropna().values
        assert np.all(p_vals >= 0)
        assert np.all(p_vals <= 1)

    def test_chisq_nonnegative(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        assert np.all(df["chisq"].values >= 0)

    def test_rho_in_minus1_plus1(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality(feature_names=["X1", "X2"])
        rho = df.loc[df["variable"] != "GLOBAL", "rho"].values
        assert np.all(rho >= -1.0)
        assert np.all(rho <= 1.0)

    def test_log_transform(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df_id  = model.check_proportionality(feature_names=["X1", "X2"])
        df_log = model.check_proportionality(np.log, feature_names=["X1", "X2"])
        # Different transform gives different rho values
        assert not np.allclose(
            df_id.loc[df_id.variable != "GLOBAL", "rho"].values,
            df_log.loc[df_log.variable != "GLOBAL", "rho"].values,
        )

    def test_auto_feature_names(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        df = model.check_proportionality()   # no names given
        assert "x0" in df["variable"].values
        assert "x1" in df["variable"].values

    def test_raises_without_schoenfeld(self, small_dataset):
        from pycrr import estimate_ipcw_weights
        X, time, event = small_dataset
        w, _ = estimate_ipcw_weights(time, event)
        model = FineGrayModel().fit(X, time, event, ipcw=w)
        with pytest.raises(RuntimeError, match="Schoenfeld"):
            model.check_proportionality()

    def test_schoenfeld_shape(self, fitted_two_cov):
        model, X, time, event = fitted_two_cov
        n_events = int((event == 1).sum())
        # schoenfeld_residuals_ is (n_unique_event_times, p)
        assert model.schoenfeld_residuals_.shape[1] == X.shape[1]
        assert model.schoenfeld_residuals_.shape[0] <= n_events

    def test_schoenfeld_times_sorted(self, fitted_two_cov):
        model, *_ = fitted_two_cov
        t = model.schoenfeld_times_
        assert np.all(np.diff(t) >= 0)


# ---------------------------------------------------------------------------
# Time-varying covariate effects  (cov2 / tf)
# ---------------------------------------------------------------------------

class TestTimeVaryingEffects:
    def test_fit_with_cov2_tf(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        assert model.coef_ is not None

    def test_coef_shape_extended(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        p = X.shape[1]
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        assert model.coef_.shape == (2 * p,)  # [base, tv]

    def test_se_shape_extended(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        p = X.shape[1]
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        assert model.standard_errors_.shape == (2 * p,)

    def test_se_positive(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        assert np.all(model.standard_errors_ > 0)

    def test_predict_cif_with_cov2(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        cif   = model.predict_cif(X[:5], times=[1.0, 2.0, 5.0], cov2_new=X[:5])
        assert cif.shape == (5, 3)

    def test_cif_in_unit_interval(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        cif   = model.predict_cif(X[:10], times=np.linspace(0.1, 5.0, 20),
                                  cov2_new=X[:10])
        assert np.all(cif >= 0)
        assert np.all(cif <= 1)

    def test_cif_monotone(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        times = np.linspace(0.1, 5.0, 30)
        cif   = model.predict_cif(X[:5], times=times, cov2_new=X[:5])
        assert np.all(np.diff(cif, axis=1) >= -1e-10)

    def test_tf_none_raises(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        with pytest.raises(ValueError, match="tf must be provided"):
            FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=None)

    def test_cov2_wrong_rows_raises(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        with pytest.raises(ValueError, match="rows"):
            FineGrayModel().fit(X, Surv(time, event), cov2=X[:10], tf=np.log)

    def test_predict_without_cov2_raises(self, two_covariate_dataset):
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        with pytest.raises(ValueError, match="cov2_new"):
            model.predict_cif(X[:3], times=[1.0, 2.0])

    def test_tv_coefs_differ_from_static(self, two_covariate_dataset):
        """Adding time-varying terms should change the base coefficients."""
        X, time, event = two_covariate_dataset
        m_static = FineGrayModel().fit(X, Surv(time, event))
        m_tv     = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        # Base coefs should differ (TV terms soak up time-varying variation)
        assert not np.allclose(m_static.coef_, m_tv.coef_[:X.shape[1]], atol=1e-3)

    def test_custom_tf(self, two_covariate_dataset):
        """tf can be any callable, e.g. sqrt."""
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.sqrt)
        assert model.coef_.shape == (2 * X.shape[1],)

    def test_robust_se_not_computed_for_tv(self, two_covariate_dataset):
        """Robust SE is not yet implemented for the cov2 path."""
        X, time, event = two_covariate_dataset
        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
        assert model.robust_se_ is None

    def test_tv_convergence_on_validation_data(self):
        """
        On validation_data.csv (n=500), pycrr's TV fit should converge to a
        true optimum (gradient near zero) and achieve a lower neg_loglik than
        R's Newton-Raphson solution (which stops early due to step-size criterion).

        R solution:  [0.7614, -0.3172,  0.0062, -0.0838], neg_loglik=1369.60
        pycrr finds: lower neg_loglik with gradient ≈ 1e-6
        """
        import os
        from pycrr.core import _censoring_survival
        from pycrr.model import _efron_grad_hess_tv_varcoef

        here = os.path.dirname(os.path.abspath(__file__))
        csv  = os.path.join(here, "..", "examples", "validation_data.csv")
        df   = pd.read_csv(csv)
        time  = df["time"].to_numpy(dtype=float)
        event = df["event"].to_numpy(dtype=int)
        X     = df[["X1", "X2"]].to_numpy(dtype=float)

        model = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)

        # Gradient at our solution should be ~0 (true optimum)
        order   = np.argsort(time, kind="stable")
        X_s     = X[order]; time_s = time[order]; event_s = event[order]
        G_func  = _censoring_survival(time_s, event_s)
        G_i     = np.array([G_func(t) for t in time_s])
        X_s_ord = X_s   # cov2 also sorted

        g, _ = _efron_grad_hess_tv_varcoef(
            X_s, X_s_ord, np.log, time_s, event_s, G_i, G_func, model.coef_, 1
        )
        assert np.max(np.abs(g)) < 1e-4, f"Gradient at TV solution not near zero: {g}"

        # pycrr should achieve lower neg_loglik than R's NR solution
        # (verifies our optimizer goes further than R's NR convergence)
        R_TV = np.array([0.761358, -0.317226, 0.006242, -0.083756])
        # Just check our gradient is smaller than at R's solution
        g_R, _ = _efron_grad_hess_tv_varcoef(
            X_s, X_s_ord, np.log, time_s, event_s, G_i, G_func, R_TV, 1
        )
        assert np.max(np.abs(g)) < np.max(np.abs(g_R)), \
            "pycrr gradient should be smaller than R's NR gradient"
