"""
Plotting utilities for competing risks analysis.
"""

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_cif(aj, cause, times=None, ax=None, ci=True, label=None, **kwargs):
    """
    Plot a cumulative incidence function with optional confidence band.

    Parameters
    ----------
    aj : AalenJohansen
        Fitted estimator.
    cause : int
        Cause to plot.
    times : array-like or None
        Evaluation grid. If None, uses all observed event times.
    ax : matplotlib Axes or None
        Axes to draw on. Creates a new figure if None.
    ci : bool
        Whether to draw the pointwise confidence band. Default True.
    label : str or None
        Legend label for the CIF line.
    **kwargs
        Passed to ``ax.step()`` for the CIF line.

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots()

    t_grid, cif = aj.predict(cause, times=times)

    kw = dict(where="post", label=label or f"Cause {cause}")
    kw.update(kwargs)
    ax.step(t_grid, cif, **kw)

    if ci:
        lo, hi = aj.confidence_intervals(cause, times=t_grid)
        color  = ax.lines[-1].get_color()
        ax.fill_between(t_grid, lo, hi, step="post", alpha=0.15, color=color)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative incidence")
    ax.set_ylim(0, None)
    ax.set_xlim(left=0)

    return ax


def plot_cif_stack(aj, causes=None, times=None, labels=None,
                   colors=None, ax=None, show_censored_band=True):
    """
    Stacked cumulative incidence plot for all competing causes.

    Displays each cause as a filled area stacked on top of the previous,
    so that the total height at any time equals the overall probability of
    having experienced any event by that time.  The remaining gap to 1 is
    the event-free probability.

    Parameters
    ----------
    aj : AalenJohansen
        Fitted estimator.
    causes : list of int or None
        Causes to include.  Defaults to all fitted causes.
    times : array-like or None
        Evaluation grid.  If None, uses all observed event times.
    labels : list of str or None
        Legend labels, one per cause.
    colors : list or None
        Colors, one per cause.  Defaults to the matplotlib color cycle.
    ax : matplotlib Axes or None
    show_censored_band : bool
        If True, draws the event-free probability as a shaded area at the top.
        Default True.

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots()

    causes = causes if causes is not None else aj.causes_

    # Build a common time grid
    if times is None:
        t_grid = aj._tables[causes[0]]["times"]
        for k in causes[1:]:
            t_grid = np.union1d(t_grid, aj._tables[k]["times"])
    else:
        t_grid = np.asarray(times, dtype=float)

    # CIF for each cause on the shared grid
    cif_matrix = []
    for k in causes:
        _, cif_k = aj.predict(k, times=t_grid)
        cif_matrix.append(cif_k)
    cif_matrix = np.array(cif_matrix)     # shape (n_causes, T)

    # Default colors from matplotlib cycle
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors     = [p["color"] for p in prop_cycle]
    if labels is None:
        labels = [f"Cause {k}" for k in causes]

    # Stacked plot: each row of cif_matrix is one layer
    bottom = np.zeros(len(t_grid))

    for i, (k, label, color) in enumerate(zip(causes, labels, colors)):
        top = bottom + cif_matrix[i]
        ax.fill_between(t_grid, bottom, top, step="post",
                        label=label, color=color, alpha=0.75)
        ax.step(t_grid, top, where="post", color=color, linewidth=0.8)
        bottom = top

    # Event-free probability band (gap to 1)
    if show_censored_band:
        ax.fill_between(t_grid, bottom, 1.0, step="post",
                        label="Event-free", color="lightgray", alpha=0.5)
        ax.step(t_grid, np.ones(len(t_grid)), where="post",
                color="gray", linewidth=0.6, linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.legend(loc="upper left")

    return ax


def plot_brier_score(eval_times, bs, ax=None, label="Brier score", **kwargs):
    """
    Plot the Brier score curve over time.

    Parameters
    ----------
    eval_times : array-like
    bs : array-like
        Brier score values at each time point.
    ax : matplotlib Axes or None
    label : str
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots()

    kw = dict(label=label)
    kw.update(kwargs)
    ax.plot(eval_times, bs, **kw)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Null (0.25)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Brier score")
    ax.set_ylim(bottom=0)
    ax.legend()

    return ax
