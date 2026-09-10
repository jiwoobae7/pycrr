import numpy as np
from lifelines import KaplanMeierFitter


def _censoring_survival(time, event, clip=1e-10):
    """
    KM estimate of the censoring survival function G(t) = P(C > t).

    Returns a callable G_func(t) -> float that evaluates the left-continuous
    step function at any time point.  Used internally by FineGrayModel for
    time-varying IPCW weights.
    """
    time  = np.asarray(time, dtype=float)
    event = np.asarray(event)
    km    = KaplanMeierFitter()
    km.fit(durations=time, event_observed=(event == 0).astype(int))

    _times = km.event_table.index.values
    _surv  = km.survival_function_["KM_estimate"].values

    def G_func(t):
        idx = int(np.clip(
            np.searchsorted(_times, np.asarray(t) - 1e-8, side="right") - 1,
            0, len(_surv) - 1,
        ))
        return float(max(_surv[idx], clip))

    return G_func


def estimate_ipcw_weights(time, event, clip=0.05):
    """
    Estimate inverse probability of censoring weights (IPCW) via Kaplan-Meier.

    Fits a KM estimator on the censoring distribution G(t) and returns
    subject-level weights 1/G(t_i-), normalized to mean 1.  Competing-event
    subjects always receive weight 1.0 (they are not censored with respect to
    the subdistribution).

    Parameters
    ----------
    time : array-like of shape (n,)
        Observed times (time to event or censoring).
    event : array-like of shape (n,)
        Event indicators: 0 = censored, 1 = event of interest, 2 = competing event.
    clip : float
        Floor applied to G(t) before inversion to prevent extreme weights.
        Default 0.05.

    Returns
    -------
    weights : ndarray of shape (n,)
        Subject-level IPCW weights, normalized so that mean(weights) = 1.
    ipcw_func : callable
        A function ``ipcw_func(t) -> float`` that evaluates the weight at any
        time point.  Useful for time-varying weight schemes.
    """
    time  = np.asarray(time,  dtype=float)
    event = np.asarray(event)

    censoring_indicator = (event == 0).astype(int)
    km = KaplanMeierFitter()
    km.fit(durations=time, event_observed=censoring_indicator)

    event_times = km.event_table.index.values
    surv_probs  = km.survival_function_["KM_estimate"].values

    def ipcw_func(t):
        idx = np.searchsorted(event_times, np.asarray(t) - 1e-8, side="right") - 1
        idx = np.clip(idx, 0, len(surv_probs) - 1)
        G   = surv_probs[idx]
        return 1.0 / np.clip(G, clip, 1.0)

    weights          = ipcw_func(time)
    weights[event == 2] = 1.0
    weights          = weights / np.mean(weights)

    return weights, ipcw_func
