"""
Visual validation of comprsk CIF predictions against R's cmprsk::crr.

Plots four covariate profiles, overlaying R (dashed) and comprsk (solid).
If the lines are indistinguishable the implementations agree.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comprsk import FineGrayModel, Surv

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Fit comprsk
# ---------------------------------------------------------------------------
df    = pd.read_csv(os.path.join(HERE, "validation_data.csv"))
time  = df["time"].to_numpy(dtype=float)
event = df["event"].to_numpy(dtype=int)
X     = df[["X1", "X2"]].to_numpy(dtype=float)

model = FineGrayModel().fit(X, Surv(time, event))

profiles = np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5], [2.0, 1.0]])
labels   = ["X1=−1, X2=+0.5", "X1=0, X2=0",
            "X1=+1, X2=−0.5", "X1=+2, X2=+1"]

# ---------------------------------------------------------------------------
# Load R reference
# ---------------------------------------------------------------------------
r = pd.read_csv(os.path.join(HERE, "r_cif_full.csv"))
r_times = r["time"].to_numpy()

# comprsk CIF at R's exact time grid
py_cif = model.predict_cif(profiles, times=r_times)   # (4, T)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = ["steelblue", "tomato", "seagreen", "darkorchid"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
axes = axes.ravel()

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    r_col = r[f"p{i+1}"].to_numpy()

    ax.plot(r_times, r_col,    color=color, lw=2.5,
            linestyle="--", label="R (cmprsk::crr)")
    ax.plot(r_times, py_cif[i], color=color, lw=1.5,
            linestyle="-",  label="comprsk", alpha=0.85)

    max_diff = np.max(np.abs(py_cif[i] - r_col))
    ax.set_title(f"Profile {i+1}: {label}\nmax |diff| = {max_diff:.2e}",
                 fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative incidence")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

fig.suptitle("CIF validation: comprsk vs R's cmprsk::crr\n"
             "Synthetic n=500 dataset, 4 covariate profiles  |  "
             "Dashed = R,  Solid = comprsk",
             fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(HERE, "validation_cif_plot.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out}")
plt.show()
