"""
Numerical validation of comprsk against R's cmprsk::crr on three real datasets.

R reference (cmprsk 2.2-11, R 4.5.0)
--------------------------------------
All R values obtained via:
    crr(ftime=..., fstatus=..., cov1=..., failcode=1, cencode=0)

Model-based SE = sqrt(diag(solve(fit$inf)))   <- what comprsk computes
Robust SE      = sqrt(diag(fit$var))          <- Lin-Wei-Ying, not implemented

Tolerance: |diff| < 0.001 for all datasets.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comprsk import FineGrayModel, Surv

HERE = os.path.dirname(os.path.abspath(__file__))
COEF_TOL = 0.001
SE_TOL   = 0.001

all_pass = True


def _check(label, r_vals, py_vals, tol, unit=""):
    global all_pass
    print(f"\n  {label}")
    print(f"  {'':20s}  {'R (cmprsk)':>12}  {'comprsk':>12}  {'|diff|':>8}  {'pass?':>6}")
    print("  " + "-" * 64)
    for name, r_v, py_v in zip(unit, r_vals, py_vals):
        diff = abs(py_v - r_v)
        ok   = diff < tol
        all_pass = all_pass and ok
        print(f"  {name:20s}  {r_v:12.7f}  {py_v:12.7f}  {diff:8.6f}  {'OK' if ok else 'FAIL':>6}")


# ===========================================================================
# 1. MGUS2  (n=1338, from R's survival package)
# ===========================================================================
print("\n" + "=" * 70)
print("MGUS2  (n=1338)  —  cause 1 = plasma cell malignancy")
print("=" * 70)

df = pd.read_csv(os.path.join(HERE, "mgus2.csv")).dropna()
df["sex_m"] = (df["sex"] == "M").astype(int)
df["event"] = np.select([df["pstat"] == 1, df["death"] == 1], [1, 2], default=0)
time  = df["futime"].to_numpy(dtype=float)
event = df["event"].to_numpy(dtype=int)
X     = df[["age", "sex_m", "hgb", "creat", "mspike"]].to_numpy(dtype=float)
feat  = ["age", "sex_m", "hgb", "creat", "mspike"]

R_COEF_MGUS2 = np.array([-0.0175110, -0.1571334, -0.0370931, -0.3072372,  0.8828250])
R_SE_MGUS2   = np.array([ 0.0073634,  0.2052947,  0.0515538,  0.2247651,  0.1591493])

model = FineGrayModel().fit(X, Surv(time, event))
_check("Coefficients", R_COEF_MGUS2, model.coef_,            COEF_TOL, feat)
_check("Model-based SE (fit$invinf)", R_SE_MGUS2, model.standard_errors_, SE_TOL, feat)


# ===========================================================================
# 2. BMT leukemia  (n=35, ships with cmprsk)
# ===========================================================================
print("\n" + "=" * 70)
print("BMT leukemia  (n=35)  —  cause 1 = relapse, covariate = disease group")
print("=" * 70)

bmt = pd.read_csv(os.path.join(HERE, "BMT_cmprsk.csv"))
X_bmt     = bmt[["dis"]].to_numpy(dtype=float)
time_bmt  = bmt["ftime"].to_numpy(dtype=float)
event_bmt = bmt["status"].to_numpy(dtype=int)

R_COEF_BMT = np.array([0.7651703])
R_SE_BMT   = np.array([0.7123289])

model_bmt = FineGrayModel().fit(X_bmt, Surv(time_bmt, event_bmt))
_check("Coefficients",              R_COEF_BMT, model_bmt.coef_,            COEF_TOL, ["dis"])
_check("Model-based SE (fit$invinf)", R_SE_BMT, model_bmt.standard_errors_, SE_TOL,   ["dis"])


# ===========================================================================
# 3. Transplant  (n=797, liver transplant)
# ===========================================================================
print("\n" + "=" * 70)
print("Transplant  (n=797)  —  cause 1 = transplant")
print("=" * 70)
print("  Note: 'year' (1990-1999) is centered before fitting.")
print("  R's crr uses Newton-Raphson (scale-invariant); comprsk uses")
print("  L-BFGS-B which requires well-scaled inputs. Centering is")
print("  standard practice and does not change the other coefficients.")

tr = pd.read_csv(os.path.join(HERE, "transplant_numeric.csv")).dropna()
feat_tr   = ["age", "sex", "year", "abo_A", "abo_B", "abo_AB"]
X_tr      = tr[feat_tr].to_numpy(dtype=float)
# Center year so L-BFGS-B converges correctly (see note above)
X_tr[:, 2] = X_tr[:, 2] - X_tr[:, 2].mean()
time_tr   = tr["futime"].to_numpy(dtype=float)
event_tr  = tr["status_num"].to_numpy(dtype=int)

R_COEF_TR = np.array([ 0.0189593,  0.4213008, -0.0097500, -0.3899362,  0.0116330, -0.2342665])
R_SE_TR   = np.array([ 0.0126850,  0.2578433,  0.0474726,  0.2821787,  0.3628942,  0.6040141])

model_tr = FineGrayModel().fit(X_tr, Surv(time_tr, event_tr))
_check("Coefficients",              R_COEF_TR, model_tr.coef_,            COEF_TOL, feat_tr)
_check("Model-based SE (fit$invinf)", R_SE_TR, model_tr.standard_errors_, SE_TOL,   feat_tr)


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if all_pass:
    print("All checks PASSED across all three real datasets.")
    print("comprsk matches cmprsk::crr coefficients and model-based SEs")
    print("within 0.001 absolute tolerance.")
else:
    print("FAILED — one or more checks exceeded tolerance.")
    sys.exit(1)
