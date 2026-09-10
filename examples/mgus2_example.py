"""
MGUS2 competing risks example.

Dataset
-------
mgus2 (n=1,384) from R's survival package.  Patients diagnosed with
monoclonal gammopathy of undetermined significance (MGUS), followed until
plasma cell malignancy (PCM, cause 1), death without PCM (cause 2), or
censoring.

Covariates used: age, sex (M=1), haemoglobin (hgb), creatinine (creat),
M-protein spike size (mspike).

This mirrors the standard cmprsk vignette analysis.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comprsk import AalenJohansen, FineGrayModel, Surv
from comprsk.compare import gray_test
from comprsk.metrics import concordance, integrated_brier_score

# ---------------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(HERE, "mgus2.csv")).dropna()

df["sex_m"] = (df["sex"] == "M").astype(int)
df["event"] = np.select(
    [df["pstat"] == 1, df["death"] == 1],
    [1,               2],
    default=0,
)

time  = df["futime"].to_numpy(dtype=float)
event = df["event"].to_numpy(dtype=int)
X     = df[["age", "sex_m", "hgb", "creat", "mspike"]].to_numpy(dtype=float)
feat  = ["age", "sex (M)", "hgb", "creat", "mspike"]

print(f"MGUS2: n={len(df)}  "
      f"PCM={np.sum(event==1)}  "
      f"death={np.sum(event==2)}  "
      f"censored={np.sum(event==0)}\n")

# ---------------------------------------------------------------------------
# 1. Fine-Gray regression for PCM (cause 1)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Fine-Gray regression  —  cause 1 (PCM)")
print("=" * 60)

model = FineGrayModel().fit(X, Surv(time, event))
model.summary(feature_names=feat)

print(f"\nConcordance index (cause 1): {concordance(model, X, time, event):.3f}")

eval_times = np.linspace(12, 240, 40)   # 1 – 20 years in months
ibs = integrated_brier_score(model, X, time, event, eval_times)
print(f"Integrated Brier score     : {ibs:.4f}")

# ---------------------------------------------------------------------------
# 2. Non-parametric CIF (Aalen-Johansen)
# ---------------------------------------------------------------------------

aj = AalenJohansen().fit(Surv(time, event))
aj.summary(cause=1, times=[60, 120, 180, 240])

# ---------------------------------------------------------------------------
# 3. Gray's test: sex difference in PCM incidence
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Gray's test  —  sex difference in PCM (cause 1)")
print("=" * 60)

result = gray_test(Surv(time, event), df["sex_m"].to_numpy())
result.summary()

# ---------------------------------------------------------------------------
# 4. CIF prediction: age profiles
# ---------------------------------------------------------------------------

t_grid = np.linspace(0, 240, 300)

# Three age profiles, average other covariates
X_means = X.mean(axis=0)
profiles = {
    "Age 50": np.array([[50, X_means[1], X_means[2], X_means[3], X_means[4]]]),
    "Age 65": np.array([[65, X_means[1], X_means[2], X_means[3], X_means[4]]]),
    "Age 80": np.array([[80, X_means[1], X_means[2], X_means[3], X_means[4]]]),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parametric (Fine-Gray)
ax = axes[0]
for label, X_p in profiles.items():
    cif = model.predict_cif(X_p, t_grid)[0]
    ax.plot(t_grid / 12, cif, label=label)
ax.set_xlabel("Years from diagnosis")
ax.set_ylabel("Cumulative incidence")
ax.set_title("Fine-Gray CIF for PCM by age\n(sex, hgb, creat, mspike at mean)")
ax.legend()
ax.set_ylim(0, 0.5)

# Non-parametric (Aalen-Johansen) with CI
ax = axes[1]
for cause, color, label in [(1, "steelblue", "PCM"), (2, "tomato", "Death w/o PCM")]:
    t_np, cif_np = aj.predict(cause, times=t_grid)
    lo, hi       = aj.confidence_intervals(cause, times=t_grid)
    ax.plot(t_np / 12, cif_np, color=color, label=label)
    ax.fill_between(t_np / 12, lo, hi, color=color, alpha=0.15)
ax.set_xlabel("Years from diagnosis")
ax.set_ylabel("Cumulative incidence")
ax.set_title("Aalen-Johansen CIF (pooled)\nwith 95% CI")
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
out_path = os.path.join(HERE, "mgus2_cif.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
