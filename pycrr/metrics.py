"""
Prediction accuracy metrics for competing risks models.

Implements the IPCW Brier score (Graf et al. 1999, adapted for competing
risks by Schoop et al. 2011) and a concordance index based on the
subdistribution linear predictor (Wolbers et al. 2014).
"""

import numpy as np
from lifelines import KaplanMeierFitter


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(model, X, time, event, eval_times, cause=1):
    """
    IPCW Brier score for competing risks CIF predictions.

    For each evaluation time t, the Brier score measures the mean squared
    difference between predicted CIF values and observed outcomes, corrected
    for censoring via inverse probability of censoring weights (IPCW).

    Parameters
    ----------
    model : fitted FineGrayModel
        Must implement ``predict_cif(X, times)``.
    X : array-like of shape (n, p)
        Covariates (test set).
    time : array-like of shape (n,)
        Observed times (test set).
    event : array-like of shape (n,)
        Event indicators (test set): 0=censored, 1=event of interest, 2+=competing.
    eval_times : array-like of shape (T,)
        Time points at which to evaluate the Brier score.
    cause : int
        Cause of interest.  Default 1.

    Returns
    -------
    eval_times : ndarray of shape (T,)
    bs : ndarray of shape (T,)
        Brier score at each evaluation time.

    References
    ----------
    Schoop, R., Beyersmann, J., Schumacher, M., & Binder, H. (2011).
    Quantifying the predictive accuracy of time-to-event models in the
    presence of competing risks. Biometrical Journal, 53(1), 88-112.
    """
    X          = np.atleast_2d(np.asarray(X,     dtype=float))
    time       = np.asarray(time,  dtype=float)
    event      = np.asarray(event, dtype=int)
    eval_times = np.asarray(eval_times, dtype=float)
    n          = len(time)

    # CIF predictions at all evaluation times — shape (n, T)
    cif = model.predict_cif(X, times=eval_times)

    # Kaplan-Meier estimate of censoring survival G(t)
    km_cens = KaplanMeierFitter()
    km_cens.fit(durations=time, event_observed=(event == 0).astype(int))

    def G(t):
        return max(float(km_cens.predict(t)), 1e-3)

    bs = np.zeros(len(eval_times))

    for j, t in enumerate(eval_times):
        G_t = G(t)
        W   = np.zeros(n)

        for i in range(n):
            if time[i] <= t and event[i] == cause:
                # Event of interest observed before/at t
                W[i] = 1.0 / G(time[i])
            elif time[i] > t:
                # Still at risk at t (event-free or not yet censored)
                W[i] = 1.0 / G_t
            # else: censored or competing event before t -> W[i] = 0

        Y      = ((time <= t) & (event == cause)).astype(float)
        resid  = cif[:, j] - Y
        bs[j]  = np.dot(W, resid**2) / n

    return eval_times, bs


def integrated_brier_score(model, X, time, event, eval_times, cause=1):
    """
    Integrated Brier score (IBS) — time-averaged prediction error.

    Integrates the Brier score curve over ``eval_times`` using the
    trapezoidal rule, normalized by the evaluation window length.

    Parameters
    ----------
    model, X, time, event, eval_times, cause
        Same as :func:`brier_score`.

    Returns
    -------
    ibs : float
        Integrated Brier score.
    """
    eval_times, bs = brier_score(model, X, time, event, eval_times, cause=cause)
    t_range = eval_times[-1] - eval_times[0]
    if t_range <= 0:
        return float(np.mean(bs))
    return float(np.trapz(bs, eval_times) / t_range)


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------

def concordance(model, X, time, event, cause=1):
    """
    C-statistic for competing risks CIF predictions.

    Uses the linear predictor ``eta = X @ model.coef_`` as the risk score.
    For each subject i with a cause-k event, compares their risk score against
    all subjects j in the subdistribution risk set at T_i.  A pair is
    concordant if eta_i > eta_j (higher predicted risk for the subject who
    had the event first).

    Parameters
    ----------
    model : fitted FineGrayModel
    X : array-like of shape (n, p)
    time : array-like of shape (n,)
    event : array-like of shape (n,)
    cause : int
        Cause of interest.  Default 1.

    Returns
    -------
    c_index : float
        Concordance index in [0, 1].  0.5 = no discrimination.

    Notes
    -----
    Complexity is O(n * n_events).  For very large datasets (n > 10 000)
    computation may be slow.

    References
    ----------
    Wolbers, M., Blanche, P., Koller, M. T., Witteman, J. C., &
    Gerds, T. A. (2014). Concordance for prognostic models with
    competing risks. Biostatistics, 15(3), 526-539.
    """
    X     = np.atleast_2d(np.asarray(X,     dtype=float))
    time  = np.asarray(time,  dtype=float)
    event = np.asarray(event, dtype=int)

    if model.coef_ is None:
        raise RuntimeError("Model is not fitted. Call fit() first.")

    eta       = X @ model.coef_
    competing = (event != 0) & (event != cause)

    concordant = 0.0
    total      = 0.0

    for i in np.where(event == cause)[0]:
        t_i = time[i]

        # Subdistribution risk set at T_i (excluding i itself)
        in_risk = (
            (time >= t_i) | ((time < t_i) & competing)
        )
        in_risk[i] = False

        n_risk = in_risk.sum()
        if n_risk == 0:
            continue

        eta_j = eta[in_risk]
        concordant += np.sum(eta[i] > eta_j) + 0.5 * np.sum(eta[i] == eta_j)
        total      += n_risk

    return float(concordant / total) if total > 0 else np.nan
