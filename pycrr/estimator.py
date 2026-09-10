"""
Non-parametric Aalen-Johansen estimator of the cumulative incidence function.
"""

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from pycrr.surv import Surv


class AalenJohansen:
    """
    Non-parametric Aalen-Johansen estimator for competing risks.

    Estimates F_k(t) = P(T <= t, cause = k) without any covariate adjustment,
    the correct non-parametric analog of 1 - KM when competing events are present.

    Parameters
    ----------
    confidence_level : float
        Confidence level for pointwise CI. Default 0.95.

    Examples
    --------
    >>> from pycrr.estimator import AalenJohansen
    >>> aj = AalenJohansen().fit(time, event)
    >>> aj.summary(cause=1)
    >>> t, cif = aj.predict(cause=1)
    >>> lo, hi  = aj.confidence_intervals(cause=1)
    """

    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.causes_   = None
        self.n_        = None
        self._tables   = {}   # cause -> dict(times, cif, var)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, time, event=None):
        """
        Fit the Aalen-Johansen estimator.

        Parameters
        ----------
        time : array-like of shape (n,)
            Observed times.
        event : array-like of shape (n,)
            Event indicators: 0 = censored, 1, 2, ... = cause codes.

        Returns
        -------
        self
        """
        if isinstance(time, Surv):
            if event is not None:
                raise ValueError(
                    "Pass either a Surv object or separate (time, event), not both."
                )
            time, event = time
        elif event is None:
            raise ValueError("event is required when time is not a Surv object.")

        time  = np.asarray(time,  dtype=float)
        event = np.asarray(event, dtype=int)
        self.n_      = len(time)
        self.causes_ = sorted(int(c) for c in np.unique(event) if c != 0)

        if not self.causes_:
            raise ValueError("No events observed (all event codes are 0).")

        # Unique times where any event (any cause) occurs
        event_times = np.sort(np.unique(time[event != 0]))

        # Walk through event times, tracking overall KM survival S(t^-)
        S            = 1.0          # overall KM, updated after each step
        S_before     = []           # S(t^-) at each event time
        Y_vec        = []           # at-risk count at each event time
        dNk_vecs     = {k: [] for k in self.causes_}

        for t in event_times:
            Y  = int(np.sum(time >= t))                  # at risk just before t
            dN = int(np.sum((time == t) & (event != 0))) # all-cause events at t

            S_before.append(S)
            Y_vec.append(Y)

            for k in self.causes_:
                dNk_vecs[k].append(int(np.sum((time == t) & (event == k))))

            if Y > 0:
                S = S * (1.0 - dN / Y)

        S_before = np.array(S_before, dtype=float)
        Y_arr    = np.array(Y_vec,    dtype=float)

        # Build per-cause CIF and variance
        for k in self.causes_:
            dNk = np.array(dNk_vecs[k], dtype=float)

            # CIF increment: dF_k(t) = S(t^-) * dN_k(t) / Y(t)
            safe_Y = np.where(Y_arr > 0, Y_arr, np.inf)
            dF     = S_before * dNk / safe_Y

            times_full = np.concatenate([[0.0], event_times])
            cif_full   = np.concatenate([[0.0], np.cumsum(dF)])

            # Variance (Greenwood-style approximation):
            # Var[F_k(t)] ≈ sum_{s<=t} S(s^-)^2 * dN_k(s) / Y(s)^2
            var_inc  = np.where(Y_arr > 0, S_before**2 * dNk / Y_arr**2, 0.0)
            var_full = np.concatenate([[0.0], np.cumsum(var_inc)])

            self._tables[k] = dict(times=times_full, cif=cif_full, var=var_full)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, cause, times=None):
        """
        Return CIF values for a given cause.

        Parameters
        ----------
        cause : int
        times : array-like or None
            Evaluation time points. If None, returns values at all observed
            event times (plus t=0).

        Returns
        -------
        times : ndarray
        cif   : ndarray
        """
        self._check_fitted(cause)
        tbl = self._tables[cause]

        if times is None:
            return tbl["times"].copy(), tbl["cif"].copy()

        times = np.asarray(times, dtype=float)
        f     = self._step_interp(tbl["times"], tbl["cif"])
        return times, f(times)

    def confidence_intervals(self, cause, times=None):
        """
        Pointwise confidence intervals via the log-log transform.

        Parameters
        ----------
        cause : int
        times : array-like or None

        Returns
        -------
        lower, upper : ndarray  (same length as times or full time grid)
        """
        self._check_fitted(cause)
        tbl = self._tables[cause]

        if times is not None:
            times = np.asarray(times, dtype=float)
            cif   = self._step_interp(tbl["times"], tbl["cif"])(times)
            var   = self._step_interp(tbl["times"], tbl["var"])(times)
        else:
            times = tbl["times"]
            cif   = tbl["cif"]
            var   = tbl["var"]

        z  = stats.norm.ppf(1.0 - (1.0 - self.confidence_level) / 2.0)
        se = np.sqrt(np.clip(var, 0.0, np.inf))

        # log-log transform: theta = log(-log F_k)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_cif = np.where(cif > 0, np.log(cif), -700.0)
            theta   = np.where(cif > 0, np.log(-log_cif), np.inf)
            se_ll   = np.where(
                (cif > 0) & (np.abs(log_cif) > 1e-10),
                se / (cif * np.abs(log_cif)),
                np.inf,
            )

        lower = np.exp(-np.exp(theta + z * se_ll))
        upper = np.exp(-np.exp(theta - z * se_ll))

        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, cause, times=None, max_rows=20):
        """
        Print a table of CIF estimates with confidence intervals.

        Parameters
        ----------
        cause : int
        times : array-like or None
            Specific evaluation time points.  If None, evenly spaced rows
            from the observed event time grid are shown.
        max_rows : int
            Maximum table rows when times is None.
        """
        self._check_fitted(cause)
        tbl = self._tables[cause]

        if times is None:
            full_t = tbl["times"]
            idx    = np.unique(
                np.linspace(0, len(full_t) - 1, min(len(full_t), max_rows), dtype=int)
            )
            t_out = full_t[idx]
        else:
            t_out = np.asarray(times, dtype=float)

        _, cif_out = self.predict(cause, times=t_out)
        lo, hi     = self.confidence_intervals(cause, times=t_out)

        n_cause = int(np.sum(
            [1 for v in self._tables.values()
             for _ in [None]] ))  # just a placeholder, see below

        print(f"Aalen-Johansen CIF  —  cause {cause}  (n={self.n_})")
        print(f"{'Time':>10}  {'CIF':>8}  {'Lower':>8}  {'Upper':>8}")
        print("-" * 44)
        for t, c, l, u in zip(t_out, cif_out, lo, hi):
            if t == 0.0:
                continue
            print(f"{t:10.4f}  {c:8.4f}  {l:8.4f}  {u:8.4f}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_interp(x, y):
        return interp1d(
            x, y,
            kind="previous",
            bounds_error=False,
            fill_value=(y[0], y[-1]),
        )

    def _check_fitted(self, cause):
        if not self._tables:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if cause not in self._tables:
            raise ValueError(
                f"Cause {cause} not in fitted causes {self.causes_}."
            )
