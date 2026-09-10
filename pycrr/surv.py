"""
Surv: survival response object analogous to R's Surv().
"""

import numpy as np


class Surv:
    """
    Survival response object that bundles time and event arrays.

    Analogous to ``Surv(time, status)`` in R's survival package.  A ``Surv``
    object can be passed wherever ``(time, event)`` are expected:

    - ``FineGrayModel().fit(X, Surv(time, event))``
    - ``AalenJohansen().fit(Surv(time, event))``
    - ``gray_test(Surv(time, event), group)``

    Parameters
    ----------
    time : array-like of shape (n,)
        Observed times (non-negative).
    event : array-like of shape (n,)
        Event indicators: 0 = censored, 1 = cause of interest, 2+ = competing
        causes.

    Examples
    --------
    >>> from pycrr import Surv, AalenJohansen, FineGrayModel
    >>> sv = Surv(time, event)
    >>> aj = AalenJohansen().fit(sv)
    >>> model = FineGrayModel().fit(X, sv)
    """

    def __init__(self, time, event):
        self.time  = np.asarray(time,  dtype=float)
        self.event = np.asarray(event, dtype=int)

        if self.time.ndim != 1 or self.event.ndim != 1:
            raise ValueError("time and event must be 1-D arrays.")
        if self.time.shape != self.event.shape:
            raise ValueError(
                f"time and event must have the same length "
                f"(got {len(self.time)} and {len(self.event)})."
            )
        if np.any(self.time < 0):
            raise ValueError("time must be non-negative.")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        """Unpack as ``time, event = surv_obj``."""
        yield self.time
        yield self.event

    def __repr__(self):
        n       = len(self.time)
        causes  = sorted(int(c) for c in np.unique(self.event) if c != 0)
        parts   = [f"n={n}"]
        for c in causes:
            parts.append(f"cause{c}={int(np.sum(self.event == c))}")
        parts.append(f"censored={int(np.sum(self.event == 0))}")
        return "Surv(" + ", ".join(parts) + ")"

    @property
    def n_events(self):
        """Total number of events (any cause)."""
        return int(np.sum(self.event != 0))

    @property
    def causes(self):
        """Sorted list of observed cause codes (excluding 0)."""
        return sorted(int(c) for c in np.unique(self.event) if c != 0)

    def event_table(self):
        """
        Return a summary dict: {cause: n_events, 'censored': n}.

        Returns
        -------
        dict
        """
        tbl = {"censored": int(np.sum(self.event == 0))}
        for c in self.causes:
            tbl[f"cause_{c}"] = int(np.sum(self.event == c))
        return tbl
