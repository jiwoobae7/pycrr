"""
Fine & Gray (1999) competing-risks regression.

Implements the proportional subdistribution hazard model via an
IPCW-weighted partial likelihood with Breslow tie correction, matching
the default behaviour of R's cmprsk::crr().

New in this version
-------------------
- Robust sandwich SE  (Lin-Wei-Ying, matching R's fit$var)
- Per-subject score residuals  (fit$res in cmprsk)
- Schoenfeld residuals + proportional subdistribution hazards test
- Time-varying covariate effects via cov2 / tf  (matching R's crr cov2/tf)
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from pycrr.core import estimate_ipcw_weights, _censoring_survival
from pycrr.surv import Surv


# ---------------------------------------------------------------------------
# Helper: fixed-weight gradient / Hessian (Efron tie correction)
# ---------------------------------------------------------------------------

def _efron_grad_hess(X, time, event, w, beta, event_of_interest):
    """
    Gradient and Hessian of the negative IPCW-weighted partial log-likelihood
    with Efron tie correction (fixed per-subject weights).
    """
    n, p    = X.shape
    grad    = np.zeros(p)
    hess    = np.zeros((p, p))

    eta     = np.clip(X @ beta, -30, 30)
    exp_eta = np.exp(eta)

    competing = (event != 0) & (event != event_of_interest)

    for t in np.sort(np.unique(time[event == event_of_interest])):
        tie_mask  = (time == t) & (event == event_of_interest)
        risk_mask = (time >= t) | ((time < t) & competing)

        if risk_mask.sum() < 2 or tie_mask.sum() == 0:
            continue

        m        = tie_mask.sum()
        X_risk   = X[risk_mask];    exp_risk = exp_eta[risk_mask]; w_risk = w[risk_mask]
        X_tie    = X[tie_mask];     exp_tie  = exp_eta[tie_mask];  w_tie  = w[tie_mask]

        S0 = np.dot(w_risk, exp_risk)
        S1 = (w_risk * exp_risk) @ X_risk
        S2 = (X_risk.T * (w_risk * exp_risk)) @ X_risk

        T0 = np.dot(w_tie, exp_tie)
        T1 = (w_tie * exp_tie) @ X_tie
        T2 = (X_tie.T * (w_tie * exp_tie)) @ X_tie

        grad += (w_tie[:, None] * X_tie).sum(axis=0)

        for l in range(m):
            frac  = l / m
            denom = max(S0 - frac * T0, 1e-12)
            numer = S1 - frac * T1
            grad -= numer / denom
            hess += (S2 - frac * T2) / denom - np.outer(numer, numer) / denom**2

    return -grad, -hess


# ---------------------------------------------------------------------------
# Helper: time-varying IPCW gradient / Hessian (Breslow tie correction)
# ---------------------------------------------------------------------------

def _efron_grad_hess_tv(X, time, event, G_i, G_func, beta, event_of_interest):
    """
    Gradient and Hessian of the negative Fine-Gray partial log-likelihood
    with time-varying IPCW weights and Breslow tie correction.

    Competing-event subjects in the subdistribution risk set at event time t
    receive weight G(t^-)/G(T_i^-) <= 1, matching cmprsk::crr.
    """
    n, p = X.shape
    grad = np.zeros(p)
    hess = np.zeros((p, p))

    eta       = np.clip(X @ beta, -30, 30)
    exp_eta   = np.exp(eta)
    competing = (event != 0) & (event != event_of_interest)

    for t in np.sort(np.unique(time[event == event_of_interest])):
        tie_mask    = (time == t) & (event == event_of_interest)
        comp_before = (time < t) & competing
        risk_mask   = (time >= t) | comp_before

        if risk_mask.sum() < 2 or not tie_mask.any():
            continue

        G_t  = G_func(t)
        w_tv = np.ones(n)
        if comp_before.any():
            G_safe             = np.where(G_i[comp_before] > 0, G_i[comp_before],
                                          np.finfo(float).tiny)
            w_tv[comp_before]  = G_t / G_safe

        m        = int(tie_mask.sum())
        X_risk   = X[risk_mask]
        wr_exp   = w_tv[risk_mask] * exp_eta[risk_mask]

        S0 = wr_exp.sum()
        S1 = (wr_exp[:, None] * X_risk).sum(axis=0)
        S2 = (X_risk.T * wr_exp) @ X_risk

        denom  = max(S0, 1e-12)
        X_tie  = X[tie_mask]
        grad  += X_tie.sum(axis=0) - m * S1 / denom
        hess  += m * (S2 / denom - np.outer(S1, S1) / denom**2)

    return -grad, -hess


# ---------------------------------------------------------------------------
# NEW: sufficient statistics reused by score residuals and Schoenfeld test
# ---------------------------------------------------------------------------

def _compute_fg_statistics(X, time, event, G_i, G_func, beta, event_of_interest):
    """
    Compute S0(t_k), weighted mean xbar(t_k), and event count d(t_k) at
    every unique cause-of-interest event time.

    Returns
    -------
    event_times : (n_e,)
    S0_arr      : (n_e,)    IPCW-weighted denominator
    barx_arr    : (n_e, p)  S1 / S0  — weighted mean covariate in risk set
    d_arr       : (n_e,)    number of events at each time
    """
    n, p      = X.shape
    eta       = np.clip(X @ beta, -30, 30)
    exp_eta   = np.exp(eta)
    competing = (event != 0) & (event != event_of_interest)

    event_times = np.sort(np.unique(time[event == event_of_interest]))
    n_e         = len(event_times)
    S0_arr      = np.empty(n_e)
    barx_arr    = np.empty((n_e, p))
    d_arr       = np.empty(n_e)

    for ki, t in enumerate(event_times):
        tie_mask    = (time == t) & (event == event_of_interest)
        comp_before = (time < t) & competing
        risk_mask   = (time >= t) | comp_before

        G_t  = G_func(t)
        w_tv = np.ones(n)
        if comp_before.any():
            G_safe             = np.where(G_i[comp_before] > 0, G_i[comp_before],
                                          np.finfo(float).tiny)
            w_tv[comp_before]  = G_t / G_safe

        wr_exp       = w_tv[risk_mask] * exp_eta[risk_mask]
        S0           = float(wr_exp.sum())
        S1           = (wr_exp[:, None] * X[risk_mask]).sum(axis=0)
        S0_arr[ki]   = max(S0, 1e-12)
        barx_arr[ki] = S1 / S0_arr[ki]
        d_arr[ki]    = int(tie_mask.sum())

    return event_times, S0_arr, barx_arr, d_arr


# ---------------------------------------------------------------------------
# NEW: per-subject score residuals  (Lin-Wei-Ying sandwich meat)
# ---------------------------------------------------------------------------

def _score_residuals_fg(X, time, event, G_i, G_func, beta, event_of_interest):
    """
    Per-subject score residuals for the Fine-Gray IPCW partial likelihood.

    For subject i:

        U_i = Delta_i (x_i - xbar(T_i))                               [event term]
            - exp(eta_i) * SUM_{t_k <= T_i} c_k (x_i - xbar(t_k))    [phase-a drag]
            - exp(eta_i)/G(T_i-) * SUM_{t_k > T_i} G(t_k-) c_k (x_i - xbar(t_k))
                                                                        [phase-b, competing only]

    where c_k = d_k / S0(t_k).

    Phase-a and phase-b drags are computed via vectorised prefix / suffix
    cumulative sums — O(n_events) precomputation, O(n) per-subject lookup.

    The robust sandwich variance is then:
        Var_robust = A^{-1} (SUM_i U_i U_i^T) A^{-1}

    matching R cmprsk fit$var (Lin & Wei 1989, as applied in Fine & Gray 1999).

    Returns
    -------
    U : (n, p)
    """
    n, p      = X.shape
    eta       = np.clip(X @ beta, -30, 30)
    exp_eta   = np.exp(eta)
    competing  = (event != 0) & (event != event_of_interest)
    cause_mask = (event == event_of_interest)

    event_times, S0_arr, barx_arr, d_arr = _compute_fg_statistics(
        X, time, event, G_i, G_func, beta, event_of_interest
    )
    n_e = len(event_times)
    c   = d_arr / S0_arr                     # (n_e,)

    # --- prefix sums for phase-a (t_k <= T_i, weight = 1) ---
    # prefix_c[j]  = sum(c[0 : j])
    # prefix_cx[j] = sum_{k < j} c[k] * barx[k]
    prefix_c  = np.concatenate([[0.0], np.cumsum(c)])
    prefix_cx = np.vstack([np.zeros((1, p)),
                            np.cumsum(c[:, None] * barx_arr, axis=0)])

    # --- suffix sums for phase-b (t_k > T_i, competing subjects only) ---
    G_at_evts = np.array([G_func(t) for t in event_times])   # (n_e,)
    Gc        = G_at_evts * c                                  # (n_e,)
    Gcx       = G_at_evts[:, None] * c[:, None] * barx_arr   # (n_e, p)
    suffix_Gc  = np.concatenate([np.cumsum(Gc[::-1])[::-1],  [0.0]])
    suffix_Gcx = np.vstack([np.cumsum(Gcx[::-1], axis=0)[::-1], np.zeros((1, p))])

    U = np.zeros((n, p))

    # term 1: event contribution (+x_i - xbar(T_i) for cause-1 subjects)
    cause_idx = np.where(cause_mask)[0]
    ki_arr    = np.clip(
        np.searchsorted(event_times, time[cause_idx]), 0, n_e - 1
    )
    U[cause_idx] += X[cause_idx] - barx_arr[ki_arr]

    # term 2: phase-a drag — all subjects, event times t_k <= T_i
    # Ka[i] = number of event times <= T_i = first suffix index for t_k > T_i
    Ka     = np.searchsorted(event_times, time, side='right')   # (n,)
    drag_a = X * prefix_c[Ka][:, None] - prefix_cx[Ka]          # (n, p)
    U     -= exp_eta[:, None] * drag_a

    # term 3: phase-b drag — competing subjects only, event times t_k > T_i
    comp_idx = np.where(competing)[0]
    if len(comp_idx) > 0:
        Kb     = Ka[comp_idx]
        G_safe = np.where(G_i[comp_idx] > 0, G_i[comp_idx], np.finfo(float).tiny)
        drag_b = X[comp_idx] * suffix_Gc[Kb][:, None] - suffix_Gcx[Kb]
        U[comp_idx] -= (exp_eta[comp_idx] / G_safe)[:, None] * drag_b

    return U


# ---------------------------------------------------------------------------
# NEW: Schoenfeld residuals (one per unique event time)
# ---------------------------------------------------------------------------

def _schoenfeld_residuals_fg(X, time, event, barx_arr, event_times,
                              event_of_interest):
    """
    Schoenfeld residuals for testing proportional subdistribution hazards.

    r_k = mean(x_i | event i at t_k) - xbar(t_k)

    One residual vector per unique event time.

    Returns
    -------
    resid : (n_e, p)
    """
    n_e   = len(event_times)
    p     = X.shape[1]
    resid = np.zeros((n_e, p))
    for ki, t in enumerate(event_times):
        mask = (time == t) & (event == event_of_interest)
        if mask.any():
            resid[ki] = X[mask].mean(axis=0) - barx_arr[ki]
    return resid


# ---------------------------------------------------------------------------
# NEW: time-varying covariate effects (cov2 / tf, matching R's crr)
# ---------------------------------------------------------------------------

def _efron_grad_hess_tv_varcoef(
    X, cov2, tf, time, event, G_i, G_func, beta, event_of_interest
):
    """
    Gradient and Hessian for Fine-Gray with time-varying covariate effects.

    At each event time t, the effective covariate for subject i is:
        x_aug_i(t) = [x_i ;  cov2_i * tf(t)]

    This matches R's crr(cov1=X, cov2=cov2, tf=tf) interface.

    Parameters
    ----------
    X    : (n, p)   baseline (time-constant) covariates
    cov2 : (n, q)   covariates with time-varying effects
    tf   : callable  t -> array-like (q,)  — time function applied element-wise
    beta : (p+q,)   extended parameter vector [beta_base ; beta_tv]

    Returns
    -------
    grad : (p+q,)
    hess : (p+q, p+q)
    """
    n, p      = X.shape
    q         = cov2.shape[1]
    ptot      = p + q
    beta_base = beta[:p]
    beta_tv   = beta[p:]

    grad      = np.zeros(ptot)
    hess      = np.zeros((ptot, ptot))
    competing = (event != 0) & (event != event_of_interest)

    for t in np.sort(np.unique(time[event == event_of_interest])):
        tie_mask    = (time == t) & (event == event_of_interest)
        comp_before = (time < t) & competing
        risk_mask   = (time >= t) | comp_before

        if risk_mask.sum() < 2 or not tie_mask.any():
            continue

        # Augmented covariate at event time t
        tf_t     = np.atleast_1d(np.asarray(tf(t), dtype=float))
        Xcov2_t  = cov2 * tf_t[None, :]                    # (n, q)
        X_aug    = np.hstack([X, Xcov2_t])                  # (n, p+q)

        eta      = np.clip(X @ beta_base + Xcov2_t @ beta_tv, -30, 30)
        exp_eta  = np.exp(eta)

        # Time-varying IPCW weights
        G_t  = G_func(t)
        w_tv = np.ones(n)
        if comp_before.any():
            G_safe             = np.where(G_i[comp_before] > 0, G_i[comp_before],
                                          np.finfo(float).tiny)
            w_tv[comp_before]  = G_t / G_safe

        m        = int(tie_mask.sum())
        X_risk   = X_aug[risk_mask]
        wr_exp   = w_tv[risk_mask] * exp_eta[risk_mask]

        S0 = wr_exp.sum()
        S1 = (wr_exp[:, None] * X_risk).sum(axis=0)        # (ptot,)
        S2 = (X_risk.T * wr_exp) @ X_risk                  # (ptot, ptot)

        denom  = max(S0, 1e-12)
        X_tie  = X_aug[tie_mask]
        # Breslow tie correction (matches cmprsk::crr)
        grad  += X_tie.sum(axis=0) - m * S1 / denom
        hess  += m * (S2 / denom - np.outer(S1, S1) / denom**2)

    return -grad, -hess


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FineGrayModel:
    """
    Fine & Gray (1999) competing-risks regression.

    Models the subdistribution hazard for a single event of interest.
    Estimation uses an IPCW-weighted partial log-likelihood with Breslow
    tie correction, maximised via L-BFGS-B — matching the default behaviour
    of R's cmprsk::crr().

    After fitting, the model exposes:

    - ``coef_``, ``standard_errors_`` — model-based (matches fit$invinf)
    - ``robust_se_``                  — Lin-Wei-Ying sandwich SE (matches fit$var)
    - ``score_residuals_``            — per-subject score residuals (matches fit$res)
    - ``schoenfeld_residuals_``       — per-event Schoenfeld residuals
    - ``check_proportionality()``     — test for constant subdistribution HR

    Parameters
    ----------
    l2 : float
        L2 (ridge) regularisation on beta. Default 0 (no regularisation).
    """

    def __init__(self, l2=0.0):
        self.l2                    = l2
        self.coef_                 = None
        self.standard_errors_      = None
        self.robust_se_            = None
        self.score_residuals_      = None
        self.schoenfeld_residuals_ = None
        self.schoenfeld_times_     = None
        self.event_of_interest     = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(X, time, event):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        X     = np.atleast_2d(np.asarray(X, dtype=np.float64))
        time  = np.asarray(time,  dtype=np.float64)
        event = np.asarray(event, dtype=np.int32)
        return X, time, event

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, time, event=None, event_of_interest=1, ipcw=None,
            cov2=None, tf=None):
        """
        Fit the Fine & Gray model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariate matrix.  Standardise before calling if covariates
            are on very different scales.
        time : array-like of shape (n,) or Surv
            Observed times (or a Surv object).
        event : array-like of shape (n,) or None
            Event indicators: 0 = censored, 1 = event of interest,
            2 = competing event.  Omit when ``time`` is a Surv object.
        event_of_interest : int
            Event code to model.  Default 1.
        ipcw : array-like of shape (n,) or None
            Pre-computed fixed IPCW weights.  When None (default), weights
            are estimated automatically via Kaplan-Meier of censoring.
        cov2 : array-like of shape (n, q) or None
            Covariates for time-varying effects, as in R's crr(cov2=...).
            When supplied, ``tf`` must also be provided.
        tf : callable or None
            Time function t -> array-like (q,).  Applied element-wise to
            ``cov2`` at each event time: effective covariate at time t is
            ``[x_i ; cov2_i * tf(t)]``.  E.g. ``tf=np.log``.

        Returns
        -------
        self
        """
        # --- Surv interface ---
        if isinstance(time, Surv):
            if event is not None:
                raise ValueError(
                    "Pass either a Surv object or separate (time, event), not both."
                )
            time, event = time
        elif event is None:
            raise ValueError("event is required when time is not a Surv object.")

        X, time, event = self._coerce(X, time, event)
        n, p = X.shape
        self.event_of_interest = event_of_interest

        # --- Validate cov2 / tf ---
        has_tv_coef = (cov2 is not None)
        if has_tv_coef:
            if tf is None:
                raise ValueError("tf must be provided when cov2 is given.")
            cov2 = np.atleast_2d(np.asarray(cov2, dtype=np.float64))
            if cov2.shape[0] != n:
                raise ValueError(f"cov2 must have {n} rows, got {cov2.shape[0]}")
            q    = cov2.shape[1]
            ptot = p + q
        else:
            q    = 0
            ptot = p

        # --- Censoring weights ---
        if ipcw is None:
            G_func          = _censoring_survival(time, event)
            G_i             = np.array([G_func(t) for t in time])
            self.ipcw_func_ = G_func
            w               = None
            use_tv          = True
        else:
            w               = np.asarray(ipcw, dtype=float).copy()
            G_func          = None
            G_i             = None
            self.ipcw_func_ = None
            use_tv          = False
            zero_mask = (event == event_of_interest) & (w == 0.0)
            if zero_mask.any():
                w[zero_mask] = 1e-3

        # --- Sort by time ---
        order  = np.argsort(time, kind="stable")
        X      = X[order];     time  = time[order];  event = event[order]
        if use_tv:
            G_i = G_i[order]
        else:
            w   = w[order]
        if has_tv_coef:
            cov2 = cov2[order]

        # Store training data
        self._X      = X
        self._time   = time
        self._event  = event
        self._w      = w
        self._G_i    = G_i
        self._G_func = G_func
        self._use_tv = use_tv
        self._cov2   = cov2
        self._tf     = tf

        competing = (event != 0) & (event != event_of_interest)

        # --- Negative partial log-likelihood ---
        def neg_loglik(beta):
            if has_tv_coef:
                b_base = beta[:p];  b_tv = beta[p:]
            else:
                b_base = beta
            eta_base = np.clip(X @ b_base, -30, 30)
            ll = 0.0

            for t in np.unique(time[event == event_of_interest]):
                tie_mask    = (time == t) & (event == event_of_interest)
                comp_before = (time < t) & competing
                risk_mask   = (time >= t) | comp_before
                if risk_mask.sum() < 2:
                    continue

                m = int(tie_mask.sum())

                if has_tv_coef:
                    tf_t   = np.atleast_1d(np.asarray(tf(t), dtype=float))
                    Xc_t   = cov2 * tf_t[None, :]
                    eta    = np.clip(eta_base + Xc_t @ b_tv, -30, 30)
                else:
                    eta = eta_base

                exp_eta = np.exp(eta)

                if use_tv:
                    G_t  = G_func(t)
                    w_tv = np.ones(n)
                    if comp_before.any():
                        G_safe             = np.where(G_i[comp_before] > 0,
                                                      G_i[comp_before], np.finfo(float).tiny)
                        w_tv[comp_before]  = G_t / G_safe
                    S0 = np.dot(w_tv[risk_mask], exp_eta[risk_mask])
                    # Log-contribution of event subjects (weight = 1)
                    if has_tv_coef:
                        ll += (eta_base[tie_mask] + (Xc_t[tie_mask] @ b_tv)).sum()
                    else:
                        ll += eta[tie_mask].sum()
                    ll -= m * np.log(max(S0, 1e-12))   # Breslow
                else:
                    S0 = np.dot(w[risk_mask], exp_eta[risk_mask])
                    T  = np.dot(w[tie_mask],  exp_eta[tie_mask])
                    ll += np.dot(w[tie_mask], eta[tie_mask])
                    for l in range(m):
                        ll -= np.log(max(S0 - (l / m) * T, 1e-12))

            reg = self.l2 * np.dot(beta, beta) if self.l2 > 0 else 0.0
            return -ll + reg

        def gradient(beta):
            if has_tv_coef:
                g, _ = _efron_grad_hess_tv_varcoef(
                    X, cov2, tf, time, event, G_i, G_func,
                    beta, event_of_interest
                )
            elif use_tv:
                g, _ = _efron_grad_hess_tv(
                    X, time, event, G_i, G_func, beta, event_of_interest
                )
            else:
                g, _ = _efron_grad_hess(X, time, event, w, beta, event_of_interest)
            if self.l2 > 0:
                g = g + 2.0 * self.l2 * beta
            return g

        result = minimize(
            fun=neg_loglik,
            x0=np.zeros(ptot),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            warnings.warn(f"Optimisation may not have converged: {result.message}")

        beta = result.x

        # --- Model-based SE via observed Fisher information ---
        if has_tv_coef:
            _, hess = _efron_grad_hess_tv_varcoef(
                X, cov2, tf, time, event, G_i, G_func, beta, event_of_interest
            )
        elif use_tv:
            _, hess = _efron_grad_hess_tv(
                X, time, event, G_i, G_func, beta, event_of_interest
            )
        else:
            _, hess = _efron_grad_hess(X, time, event, w, beta, event_of_interest)

        ridge = 1e-6 * max(float(np.trace(-hess)), 1.0)
        try:
            cov = np.linalg.inv(-hess + ridge * np.eye(ptot))
        except np.linalg.LinAlgError:
            cov = np.full((ptot, ptot), np.nan)

        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        self.coef_            = beta
        self.standard_errors_ = se
        self.cov_matrix_      = cov
        z                     = beta / np.where(se > 0, se, np.nan)
        self.z_scores_        = z
        self.p_values_        = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
        self.ci_lower_        = beta - 1.96 * se
        self.ci_upper_        = beta + 1.96 * se

        # --- Robust sandwich SE and score residuals ---
        # Only implemented for the default time-varying IPCW path (use_tv=True)
        # and without cov2 (time-varying covariate path can be added later).
        if use_tv and not has_tv_coef:
            U = _score_residuals_fg(
                X, time, event, G_i, G_func, beta, event_of_interest
            )
            meat         = U.T @ U                     # (p, p)
            robust_cov   = cov @ meat @ cov            # sandwich A^{-1} B A^{-1}
            robust_se    = np.sqrt(np.clip(np.diag(robust_cov), 0.0, np.inf))

            self.score_residuals_   = U
            self.robust_cov_matrix_ = robust_cov
            self.robust_se_         = robust_se
            rz = beta / np.where(robust_se > 0, robust_se, np.nan)
            self.robust_z_          = rz
            self.robust_p_          = 2.0 * (1.0 - stats.norm.cdf(np.abs(rz)))

            # Schoenfeld residuals for proportionality test
            event_times, _, barx_arr, d_arr = _compute_fg_statistics(
                X, time, event, G_i, G_func, beta, event_of_interest
            )
            self.schoenfeld_times_     = event_times
            self.schoenfeld_residuals_ = _schoenfeld_residuals_fg(
                X, time, event, barx_arr, event_times, event_of_interest
            )
        else:
            self.score_residuals_      = None
            self.robust_cov_matrix_    = None
            self.robust_se_            = None
            self.robust_z_             = None
            self.robust_p_             = None
            self.schoenfeld_times_     = None
            self.schoenfeld_residuals_ = None

        return self

    # ------------------------------------------------------------------
    # Proportional subdistribution hazards test
    # ------------------------------------------------------------------

    def check_proportionality(self, time_transform=None, feature_names=None):
        """
        Test the proportional subdistribution hazards assumption.

        Regresses the Schoenfeld residuals against (transformed) event
        times.  A significant correlation indicates that the subdistribution
        hazard ratio varies with time — violating the proportionality
        assumption.

        This is the Fine-Gray analogue of the Grambsch-Therneau (1994)
        test for Cox models.

        Parameters
        ----------
        time_transform : callable or None
            Applied to event times before computing correlations.
            Default None uses identity (raw time).
            Common choices: ``np.log``, ``lambda t: np.log(t + 1)``.
        feature_names : list of str or None

        Returns
        -------
        pandas.DataFrame with columns: variable, rho, chisq, df, p
        """
        if self.schoenfeld_residuals_ is None:
            raise RuntimeError(
                "Schoenfeld residuals not available.  "
                "Refit with default ipcw=None (time-varying weights)."
            )

        event_times = self.schoenfeld_times_
        resid       = self.schoenfeld_residuals_   # (n_e, p)
        p           = resid.shape[1]

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]

        if time_transform is not None:
            t_vals = np.array([float(time_transform(t)) for t in event_times])
        else:
            t_vals = event_times.astype(float)

        valid  = np.isfinite(t_vals)
        t_vals = t_vals[valid]
        resid  = resid[valid]
        n_e    = len(t_vals)

        rows = []
        for j, name in enumerate(feature_names):
            r         = resid[:, j]
            rho, pval = stats.pearsonr(t_vals, r)
            chisq     = n_e * rho**2          # approximate chi-square(1)
            rows.append({"variable": name, "rho": rho,
                         "chisq": chisq, "df": 1, "p": pval})

        # Global test: sum of per-covariate chi-squares (approximate, df = p)
        chisq_g = sum(r["chisq"] for r in rows)
        pval_g  = 1.0 - stats.chi2.cdf(chisq_g, df=p)
        rows.append({"variable": "GLOBAL", "rho": np.nan,
                     "chisq": chisq_g, "df": p, "p": pval_g})

        tf_name = "identity" if time_transform is None else getattr(
            time_transform, "__name__", str(time_transform)
        )
        print(f"\nProportional subdistribution hazards test")
        print(f"Time transform : {tf_name}   |   n events : {n_e}")
        print(f"\n{'Variable':15s}  {'rho':>7}  {'chisq':>8}  {'df':>4}  {'p':>8}")
        print("-" * 48)
        for row in rows:
            rho_s = f"{row['rho']:7.4f}" if not np.isnan(row["rho"]) else "     --"
            print(f"{row['variable']:15s}  {rho_s}  {row['chisq']:8.3f}  "
                  f"{row['df']:4d}  {row['p']:8.4f}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Predict CIF
    # ------------------------------------------------------------------

    def predict_cif(self, X_new, times=None, cov2_new=None):
        """
        Predict the cumulative incidence function (CIF) for new samples.

        Uses a Breslow-type estimate of the cumulative baseline
        subdistribution hazard.

        Parameters
        ----------
        X_new : array-like of shape (m, p)
        times : array-like of shape (T,) or None
            Evaluation time points.  Defaults to all unique event times.
        cov2_new : array-like of shape (m, q) or None
            Required when the model was fitted with cov2/tf.

        Returns
        -------
        cif : ndarray of shape (m, T)
            cif[i, j] = P(T <= times[j], cause = event_of_interest | x_i)
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.to_numpy(dtype=np.float64)
        X_new = np.atleast_2d(np.asarray(X_new, dtype=np.float64))
        m     = X_new.shape[0]

        has_tv_coef = (self._cov2 is not None)
        p = self._X.shape[1]

        if has_tv_coef:
            if cov2_new is None:
                raise ValueError("Model was fitted with cov2/tf; supply cov2_new.")
            cov2_new = np.atleast_2d(np.asarray(cov2_new, dtype=np.float64))
            beta_base = self.coef_[:p]
            beta_tv   = self.coef_[p:]
        else:
            beta_base = self.coef_
            beta_tv   = None

        eta_new_base = X_new @ beta_base    # (m,)

        event_times = np.sort(
            np.unique(self._time[self._event == self.event_of_interest])
        )
        if times is None:
            times = np.concatenate([[0.0], event_times])
        times = np.asarray(times, dtype=float)

        eta_train  = self._X @ beta_base
        exp_eta_tr = np.exp(np.clip(eta_train, -30, 30))
        competing_tr = (self._event != 0) & (self._event != self.event_of_interest)

        # --- Build baseline cumhaz ---
        if not has_tv_coef:
            # Standard case: HR is time-constant per new subject
            h0_times = [0.0]
            H0_vals  = [0.0]

            for t in event_times:
                dN = (self._time == t) & (self._event == self.event_of_interest)
                dm = int(dN.sum())
                if dm == 0:
                    continue

                comp_before = (self._time < t) & competing_tr
                risk_mask   = (self._time >= t) | comp_before

                if self._use_tv:
                    G_t  = self._G_func(t)
                    w_tv = np.ones(len(self._time))
                    if comp_before.any():
                        G_safe             = np.where(self._G_i[comp_before] > 0,
                                                      self._G_i[comp_before],
                                                      np.finfo(float).tiny)
                        w_tv[comp_before]  = G_t / G_safe
                    S0 = np.dot(w_tv[risk_mask], exp_eta_tr[risk_mask])
                else:
                    S0 = np.dot(self._w[risk_mask], exp_eta_tr[risk_mask])

                h0_times.append(t)
                H0_vals.append(H0_vals[-1] + dm / max(S0, 1e-12))

            interp = interp1d(
                h0_times, H0_vals,
                kind="previous", bounds_error=False,
                fill_value=(0.0, H0_vals[-1]),
            )
            H0 = interp(times)

            exp_eta_new = np.exp(np.clip(eta_new_base, -30, 30))
            cif = 1.0 - np.exp(-np.outer(exp_eta_new, H0))

        else:
            # Time-varying effects: HR depends on t, so we must accumulate
            # the cumulative hazard event-time by event-time.
            cov2_tr = self._cov2

            # dH_0(t_k) for each event time
            h0_jumps = {}   # t_k -> dH_0(t_k)

            for t in event_times:
                dN = (self._time == t) & (self._event == self.event_of_interest)
                dm = int(dN.sum())
                if dm == 0:
                    continue

                comp_before = (self._time < t) & competing_tr
                risk_mask   = (self._time >= t) | comp_before

                tf_t      = np.atleast_1d(np.asarray(self._tf(t), dtype=float))
                Xc_t_tr   = cov2_tr * tf_t[None, :]
                eta_t     = eta_train + Xc_t_tr @ beta_tv

                G_t  = self._G_func(t)
                w_tv = np.ones(len(self._time))
                if comp_before.any():
                    G_safe             = np.where(self._G_i[comp_before] > 0,
                                                  self._G_i[comp_before],
                                                  np.finfo(float).tiny)
                    w_tv[comp_before]  = G_t / G_safe

                exp_eta_t = np.exp(np.clip(eta_t, -30, 30))
                S0        = np.dot(w_tv[risk_mask], exp_eta_t[risk_mask])
                h0_jumps[t] = dm / max(S0, 1e-12)

            # For each new subject, accumulate hazard accounting for time-varying HR
            sorted_et  = np.array(sorted(h0_jumps.keys()))
            dH0        = np.array([h0_jumps[t] for t in sorted_et])

            # cumhaz[i, k] = sum_{j <= k} dH0[j] * exp(eta_new_i(t_j))
            cif = np.zeros((m, len(times)))
            for i in range(m):
                eta_new_i = eta_new_base[i]
                cumhaz = 0.0
                cumhaz_at_et = []
                for ki, t_k in enumerate(sorted_et):
                    tf_t      = np.atleast_1d(np.asarray(self._tf(t_k), dtype=float))
                    hr_i_at_t = np.exp(np.clip(
                        eta_new_i + (cov2_new[i] * tf_t) @ beta_tv, -30, 30
                    ))
                    cumhaz += dH0[ki] * hr_i_at_t
                    cumhaz_at_et.append(cumhaz)

                cumhaz_at_et = np.array(cumhaz_at_et)
                # Interpolate to requested times
                interp = interp1d(
                    np.concatenate([[0.0], sorted_et]),
                    np.concatenate([[0.0], cumhaz_at_et]),
                    kind="previous", bounds_error=False,
                    fill_value=(0.0, cumhaz_at_et[-1]),
                )
                cif[i] = 1.0 - np.exp(-interp(times))

        self.baseline_cumhaz_ = (
            np.array(h0_times if not has_tv_coef else
                     [0.0] + list(sorted_et)),
            np.array(H0_vals  if not has_tv_coef else
                     [0.0] + list(np.cumsum(dH0))),
        )
        return cif

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def hazard_ratios(self):
        """Subdistribution hazard ratios: exp(coef_)."""
        return np.exp(self.coef_)

    def hazard_ratio_ci(self, level=0.95, robust=False):
        """
        Confidence intervals for subdistribution hazard ratios.

        Parameters
        ----------
        level  : float  Confidence level.  Default 0.95.
        robust : bool   If True, use robust SE.  Default False.
        """
        z  = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
        se = self.robust_se_ if robust else self.standard_errors_
        return list(zip(np.exp(self.coef_ - z * se),
                        np.exp(self.coef_ + z * se)))

    def summary(self, feature_names=None, robust=False):
        """
        Print a regression summary table.

        Parameters
        ----------
        feature_names : list of str or None
        robust : bool
            If True, use robust (Lin-Wei-Ying sandwich) SE and p-values.
            Default False (model-based SE matching R's fit$invinf).
        """
        if self.coef_ is None:
            print("Model not fitted.")
            return

        p = len(self.coef_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]

        if robust:
            if self.robust_se_ is None:
                raise RuntimeError("Robust SE not available (refit with default ipcw=None).")
            se  = self.robust_se_
            z   = self.robust_z_
            pv  = self.robust_p_
            se_label = "robust_se"
        else:
            se  = self.standard_errors_
            z   = self.z_scores_
            pv  = self.p_values_
            se_label = "se(model)"

        hr  = self.hazard_ratios()
        cis = self.hazard_ratio_ci(robust=robust)

        # Show both SEs if both are available
        show_both = (self.robust_se_ is not None) and (not robust)
        if show_both:
            header = (
                f"{'Variable':15s}  {'coef':>8}  {'HR':>8}  "
                f"{'se':>8}  {'robust_se':>9}  {'z':>7}  {'p':>8}  95% CI (model-based)"
            )
            print(header)
            print("-" * (len(header) + 4))
            for i, name in enumerate(feature_names):
                lo, hi = cis[i]
                print(
                    f"{name:15s}  {self.coef_[i]:8.4f}  {hr[i]:8.4f}  "
                    f"{self.standard_errors_[i]:8.4f}  {self.robust_se_[i]:9.4f}  "
                    f"{self.z_scores_[i]:7.3f}  {self.p_values_[i]:8.4f}  "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
        else:
            header = (
                f"{'Variable':15s}  {'coef':>8}  {'HR':>8}  "
                f"{se_label:>9}  {'z':>7}  {'p':>8}  95% CI"
            )
            print(header)
            print("-" * (len(header) + 4))
            for i, name in enumerate(feature_names):
                lo, hi = cis[i]
                print(
                    f"{name:15s}  {self.coef_[i]:8.4f}  {hr[i]:8.4f}  "
                    f"{se[i]:9.4f}  {z[i]:7.3f}  {pv[i]:8.4f}  [{lo:.4f}, {hi:.4f}]"
                )
