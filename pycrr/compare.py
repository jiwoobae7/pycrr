"""
Gray's K-sample test for comparing cumulative incidence functions.
"""

import numpy as np
from scipy import stats
from pycrr.surv import Surv


class GrayTestResult:
    """Result of Gray's test."""

    def __init__(self, statistic, pvalue, df, group_stats, groups):
        self.statistic   = statistic
        self.pvalue      = pvalue
        self.df          = df
        self.group_stats = group_stats   # {group_label: U_g}
        self.groups      = groups

    def __repr__(self):
        return (
            f"GrayTestResult(statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4f}, df={self.df})"
        )

    def summary(self):
        print(f"Gray's test  —  chi-squared = {self.statistic:.4f}, "
              f"df = {self.df}, p = {self.pvalue:.4f}")
        print(f"{'Group':>12}  {'U statistic':>12}")
        print("-" * 27)
        for g, u in self.group_stats.items():
            print(f"{str(g):>12}  {u:12.4f}")


def gray_test(time, event, group=None, cause=1, rho=0):
    """
    Gray's K-sample test for equality of cumulative incidence functions.

    Tests H0: F_k^(1)(t) = ... = F_k^(G)(t) for all t, where k is the
    specified cause of interest and superscripts denote groups.

    The test uses the subdistribution risk set (subjects who experienced
    competing events remain at risk) and optionally applies CIF-based
    weighting following Gray (1988).

    Parameters
    ----------
    time : array-like of shape (n,)
        Observed times.
    event : array-like of shape (n,)
        Event indicators: 0 = censored, 1 = cause of interest, 2+ = competing.
    group : array-like of shape (n,)
        Group labels.  Can be any comparable values (int, str, etc.).
    cause : int
        Which cause to test.  Default 1.
    rho : float
        Weight exponent for Fleming-Harrington-type weighting.
        ``rho=0`` (default) gives equal weights (log-rank type).
        ``rho=1`` downweights late differences.

    Returns
    -------
    GrayTestResult
        Attributes: ``statistic``, ``pvalue``, ``df``, ``group_stats``.

    References
    ----------
    Gray, R. J. (1988). A class of K-sample tests for comparing the
    cumulative incidence of a competing risk. Annals of Statistics,
    16(3), 1141-1154.
    """
    if isinstance(time, Surv):
        # gray_test(Surv(time, event), group, ...)
        group = event
        time, event = time
    elif group is None:
        raise ValueError("group is required when time is not a Surv object.")
    time  = np.asarray(time,  dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group)
    n     = len(time)

    groups  = np.sort(np.unique(group))
    G       = len(groups)

    if G < 2:
        raise ValueError("gray_test requires at least 2 groups.")

    # Unique times where cause-k events occur (pooled across groups)
    cause_mask  = event == cause
    cause_times = np.sort(np.unique(time[cause_mask]))

    if len(cause_times) == 0:
        raise ValueError(f"No events of cause {cause} found in the data.")

    # Pooled CIF for weight function (needed when rho > 0)
    pooled_cif_at_t = np.zeros(len(cause_times))
    if rho > 0:
        from pycrr.estimator import AalenJohansen
        aj = AalenJohansen().fit(time, event)
        if cause in aj.causes_:
            _, cif_vals = aj.predict(cause, times=cause_times)
            # We need CIF(t^-): use value at the step just before t
            # Step interp already gives left-continuous values at t via "previous"
            # Shift to get F(t^-): use interp at t - epsilon
            eps = np.finfo(float).eps * cause_times
            _, cif_before = aj.predict(cause, times=np.maximum(cause_times - 1e-8, 0))
            pooled_cif_at_t = cif_before

    # U statistics for each group, covariance matrix
    U     = np.zeros(G)
    Sigma = np.zeros((G, G))

    for j, t in enumerate(cause_times):
        # Weight at time t
        w = (max(1.0 - pooled_cif_at_t[j], 0.0)) ** rho if rho > 0 else 1.0

        # Subdistribution risk set at t (pooled):
        # subject i is at risk if T_i >= t OR (T_i < t AND cause_i is competing)
        competing = (event != 0) & (event != cause)
        in_risk   = (time >= t) | ((time < t) & competing)
        Y_total   = int(in_risk.sum())

        if Y_total == 0:
            continue

        # Total cause-k events at t
        dN_total = int(np.sum((time == t) & cause_mask))

        # Per-group at-risk counts and event counts
        Y_g  = np.array([int((in_risk & (group == g)).sum()) for g in groups])
        dN_g = np.array([int(np.sum((time == t) & cause_mask & (group == g)))
                         for g in groups])

        expected = Y_g / Y_total * dN_total
        U += w * (dN_g - expected)

        # Covariance update
        for gi in range(G):
            for hi in range(G):
                if gi == hi:
                    Sigma[gi, gi] += (
                        w**2
                        * Y_g[gi] * (Y_total - Y_g[gi])
                        / Y_total**2
                        * dN_total
                    )
                else:
                    Sigma[gi, hi] -= (
                        w**2
                        * Y_g[gi] * Y_g[hi]
                        / Y_total**2
                        * dN_total
                    )

    # Drop last group (constraint: sum U_g = 0)
    U_trunc   = U[:G - 1]
    Sig_trunc = Sigma[:G - 1, :G - 1]

    # Chi-squared statistic
    try:
        Sig_inv  = np.linalg.inv(Sig_trunc)
        stat     = float(U_trunc @ Sig_inv @ U_trunc)
        stat     = max(stat, 0.0)   # numerical guard
    except np.linalg.LinAlgError:
        stat = np.nan

    pvalue = float(1.0 - stats.chi2.cdf(stat, df=G - 1)) if np.isfinite(stat) else np.nan

    return GrayTestResult(
        statistic=stat,
        pvalue=pvalue,
        df=G - 1,
        group_stats={g: float(U[i]) for i, g in enumerate(groups)},
        groups=groups.tolist(),
    )
