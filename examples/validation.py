"""
Numerical validation of comprsk against R's cmprsk::crr.

Dataset
-------
n=500 synthetic two-cause competing risks observations generated in R with
set.seed(42).  The same CSV is loaded here so both implementations see
identical input data.

R reference (cmprsk 2.2-11, R 4.5.0)
--------------------------------------
    crr(ftime=time, fstatus=event, cov1=cbind(X1, X2), failcode=1, cencode=0)

Coefficients  (fit$coef)
    coef_X1 =  0.765965   coef_X2 = -0.378533

Standard errors — two variants:
    fit$invinf  (model-based, inverse Fisher information):
        se_X1 = 0.06916    se_X2 = 0.06012

    fit$var  (Lin-Wei-Ying robust variance, accounts for estimated IPCW weights):
        se_X1 = 0.08066    se_X2 = 0.06083

comprsk computes the model-based SE (matching fit$invinf).
The robust Lin-Wei-Ying correction is not implemented.

Tolerances
----------
Coefficients  : absolute difference < 0.001
Model-based SE: absolute difference < 0.001
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comprsk import FineGrayModel, Surv

# ---------------------------------------------------------------------------
# R reference values
# ---------------------------------------------------------------------------

# Coefficients (fit$coef)
R_COEF = np.array([0.7659650, -0.3785332])

# Model-based SE (fit$invinf = solve(fit$inf)) — this is what comprsk computes
R_SE_MODELBASED = np.array([0.06916, 0.06012])

# Robust SE (fit$var, Lin-Wei-Ying correction) — not implemented in comprsk
R_SE_ROBUST = np.array([0.08066, 0.06083])

COEF_TOL = 0.001
SE_TOL   = 0.001

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(HERE, "validation_data.csv")

df    = pd.read_csv(csv_path)
time  = df["time"].to_numpy()
event = df["event"].to_numpy().astype(int)
X     = df[["X1", "X2"]].to_numpy()

print(f"Loaded {len(df)} observations from {os.path.basename(csv_path)}")
print(f"Events: cause_1={np.sum(event==1)}, cause_2={np.sum(event==2)}, "
      f"censored={np.sum(event==0)}\n")

# ---------------------------------------------------------------------------
# Fit comprsk
# ---------------------------------------------------------------------------
model = FineGrayModel().fit(X, Surv(time, event))

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
print(f"{'':20s}  {'R (cmprsk)':>12}  {'comprsk':>12}  {'|diff|':>8}  {'pass?':>6}")
print("-" * 68)

all_pass = True

for i, name in enumerate(["X1", "X2"]):
    diff = abs(model.coef_[i] - R_COEF[i])
    ok   = diff < COEF_TOL
    all_pass = all_pass and ok
    print(f"  coef_{name}           {R_COEF[i]:12.6f}  {model.coef_[i]:12.6f}  "
          f"{diff:8.6f}  {'OK' if ok else 'FAIL':>6}")

print()
print("  Model-based SE (matches fit$invinf = solve(fit$inf)):")
for i, name in enumerate(["X1", "X2"]):
    diff = abs(model.standard_errors_[i] - R_SE_MODELBASED[i])
    ok   = diff < SE_TOL
    all_pass = all_pass and ok
    print(f"  se_{name}  (model)     {R_SE_MODELBASED[i]:12.5f}  "
          f"{model.standard_errors_[i]:12.6f}  {diff:8.6f}  {'OK' if ok else 'FAIL':>6}")

print()
print("  Robust SE (fit$var, Lin-Wei-Ying — not implemented in comprsk):")
for i, name in enumerate(["X1", "X2"]):
    print(f"  se_{name}  (robust)    {R_SE_ROBUST[i]:12.5f}  "
          f"{'N/A':>12}  {'N/A':>8}  {'N/A':>6}")

print()
if all_pass:
    print("All checks PASSED — comprsk matches cmprsk::crr coefficients and")
    print("model-based SEs within tolerance.")
else:
    print("FAILED — one or more checks exceeded tolerance.")
    sys.exit(1)
