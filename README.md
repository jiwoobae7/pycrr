# pycrr

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22685247.svg)](https://doi.org/10.5281/zenodo.22685247)

**Fine-Gray competing risks regression with diagnostic inference in Python.**

`pycrr` implements the subdistribution hazard model of Fine & Gray (1999) for
time-to-event data with competing events. Beyond coefficient estimation, it
provides a focused **inference and diagnostic workflow**: robust sandwich standard
errors, per-subject score residuals, a proportional subdistribution hazards test,
and time-varying covariate effects — matching the R `cmprsk::crr` interface and
validated numerically against it.

---

## Installation

```bash
pip install pycrr
```

Dependencies: `numpy`, `scipy`, `pandas`, `lifelines`.

---

## Quick start

```python
import numpy as np
from pycrr import FineGrayModel, Surv

# event: 0=censored, 1=event of interest, 2=competing event
model = FineGrayModel().fit(X, Surv(time, event))
model.summary(feature_names=["age", "treatment"])
```

```
Variable          coef        HR        se  robust_se        z         p  95% CI (model-based)
------------------------------------------------------------------------------------------------
age             0.0312    1.0317    0.0081     0.0094    3.852    0.0001  [0.0153, 0.0471]
treatment       0.4201    1.5222    0.1034     0.1187    4.062    0.0000  [0.2175, 0.6228]
```

---

## Diagnostic workflow

```python
# Proportional subdistribution hazards test
model.check_proportionality(feature_names=["age", "treatment"])
# Variable    rho    chisq  df       p
# age       0.031    0.423   1   0.515
# treatment 0.018    0.141   1   0.707
# GLOBAL       --    0.564   2   0.754

# Score residuals (n x p) — sum to zero at MLE, reconstruct robust covariance
U = model.score_residuals_          # shape (n, p)
print(U.sum(axis=0))                # ≈ [0, 0]

# Robust (sandwich) SE — accounts for estimated censoring weights
print(model.robust_se_)             # Lin-Wei-Ying sqrt(diag(A⁻¹ B A⁻¹))
print(model.robust_p_)              # p-values from robust SE
```

---

## Time-varying covariate effects

```python
# Matches R's crr(cov1=X, cov2=X, tf=log) interface
model_tv = FineGrayModel().fit(X, Surv(time, event), cov2=X, tf=np.log)
# Effective covariate at time t: [x_i ; x_i * log(t)]
# Detects whether the hazard ratio changes over time
```

---

## CIF prediction

```python
times = np.linspace(0, 10, 200)
cif   = model.predict_cif(X_new, times)   # shape (n_new, 200)

import matplotlib.pyplot as plt
plt.plot(times, cif[0], label="Profile 1")
plt.plot(times, cif[1], label="Profile 2")
plt.xlabel("Time"); plt.ylabel("CIF"); plt.legend()
```

---

## Full API

### `Surv(time, event)`

R-like survival object accepted by all fitting and testing functions.

```python
sv = Surv(time, event)
sv.causes          # sorted list of non-zero event codes
sv.n_events        # total non-censored count
sv.event_table()   # dict with counts per cause and censored
```

---

### `FineGrayModel(l2=0.0)`

#### `.fit(X, time, event=None, event_of_interest=1, ipcw=None, cov2=None, tf=None)`

| Parameter | Description |
|-----------|-------------|
| `X` | `(n, p)` covariate matrix. Center/scale if covariates are on different scales. |
| `time` | `(n,)` observed times, or a `Surv(time, event)` object. |
| `event` | `(n,)` event indicators. Omit if `time` is a `Surv` object. |
| `event_of_interest` | Event code to model. Default 1. |
| `ipcw` | Pre-computed static weights. If `None`, time-varying IPCW are used (recommended). |
| `cov2` | `(n, q)` covariates for time-varying effects (R's `cov2`). |
| `tf` | Function of time applied to `cov2` (e.g. `np.log`). Required if `cov2` is given. |

**Attributes after fitting:**

| Attribute | Description |
|-----------|-------------|
| `coef_` | Coefficients $\hat\beta$ |
| `standard_errors_` | Model-based SE (matches R `fit$invinf`) |
| `robust_se_` | Lin-Wei-Ying sandwich SE (matches R `fit$var`) |
| `robust_p_` | Two-sided p-values from robust SE |
| `score_residuals_` | `(n, p)` per-subject score residuals |
| `schoenfeld_residuals_` | Per-event Schoenfeld residuals |
| `cov_matrix_` | Model-based covariance (inverse information) |
| `hazard_ratios_` via `.hazard_ratios()` | `exp(coef_)` |

#### `.check_proportionality(tf=None, feature_names=None)`

Tests the proportional subdistribution hazards assumption. Returns a DataFrame
with columns `variable`, `rho`, `chisq`, `df`, `p`. Uses Schoenfeld-type
residuals correlated with event time (or `tf(time)` if provided).

#### `.predict_cif(X_new, times=None, cov2_new=None)`

Returns predicted CIF as `(n_new, len(times))` array.

#### `.summary(feature_names=None, robust=False)`

Prints a formatted regression table. Set `robust=True` to use robust SEs.

---

### `AalenJohansen()`

Non-parametric CIF estimator with log-log confidence intervals.

```python
aj = AalenJohansen().fit(Surv(time, event))
t_grid, cif = aj.predict(cause=1)
lo, hi      = aj.confidence_intervals(cause=1)
aj.summary()
```

---

### `gray_test(time, event, group, cause=1, rho=0)`

Gray's K-sample test for equality of CIFs across groups.

```python
from pycrr.compare import gray_test

result = gray_test(Surv(time, event), group)
result.summary()
# Gray's test — chi-squared = 5.12, df = 1, p = 0.024
```

---

### Metrics

```python
from pycrr.metrics import brier_score, integrated_brier_score, concordance

bs  = brier_score(model, X_test, time_test, event_test, t=5.0)
ibs = integrated_brier_score(model, X_test, time_test, event_test,
                              times=np.linspace(0.5, 8, 50))
c   = concordance(model, X_test, time_test, event_test)
```

---

## Validation

`pycrr` was validated against `cmprsk::crr` (R 4.5.0, v2.2-12) on:

**Synthetic dataset** (`n=500`, generated with `set.seed(42)` in R):

| Output | Max \|diff vs R\| |
|--------|-------------------|
| Coefficients | 2 × 10⁻⁶ |
| Model-based SE | 2 × 10⁻⁶ |
| Robust SE | 5 × 10⁻⁵ |
| Full 2×2 robust cov matrix | 1 × 10⁻⁴ |

**MGUS2** (`n=1,384`, from the R `survival` package, real clinical data):

| Output | Max \|diff vs R\| |
|--------|-------------------|
| Coefficients | 3 × 10⁻⁵ |
| Model-based SE | 3 × 10⁻⁵ |
| Robust SE | 2 × 10⁻⁵ |
| Full 2×2 robust cov matrix | 1 × 10⁻⁴ |

Score residuals satisfy $\sum_i U_i = 0$ to $10^{-11}$ and reconstruct the
robust covariance matrix exactly. The proportionality test detects crossing
subdistribution hazards (Schoenfeld ρ ≈ −0.65, p < 10⁻⁴) and correctly finds
no violation on proportional data. Full validation scripts are in `examples/`.

---

## Related packages

`pycrr` focuses on inference and diagnostics. Other Python competing-risks
packages take different approaches:

| Package | Focus |
|---------|-------|
| `comprisk` | Fine-Gray + penalized FG + cause-specific Cox + competing-risks forest + sklearn API |
| `pycmprsk` | Systematic port of R `cmprsk`, including censoring strata |
| `CompRiskReg` | Translation of R `cmprsk` with `cov2`/`tf` interface |
| `lifelines` | Broad survival analysis; Aalen-Johansen but not Fine-Gray |
| `scikit-survival` | ML-focused survival; non-parametric CIF only |

`pycrr`'s distinct focus is the **diagnostic layer**: score residuals whose
estimating equations balance, exact sandwich covariance reconstruction, and a
proportional subdistribution hazards test with both negative- and
positive-control validation.

---

## Background

Fine & Gray (1999) model the subdistribution hazard:

$$\lambda_1(t \mid x) = \lambda_{10}(t) \exp(\beta^\top x)$$

Subjects who experienced a competing event remain in the risk set with
time-varying weight $G(t^-)/G(T_i^-) \le 1$, where $G$ is the Kaplan-Meier
censoring survival function. `pycrr` implements this with a Breslow tie
correction (matching `cmprsk::crr`) and L-BFGS-B optimization with analytically
derived gradients.

---

## Best practices

- **Standardize covariates** when scales differ. R's `crr` uses Newton-Raphson
  (scale-invariant); `pycrr` uses L-BFGS-B which can be slow on poorly scaled
  inputs.
- **Use `Surv(time, event)`** to bundle outcomes and avoid argument order errors.
- **Check proportionality** before interpreting coefficients as time-constant.

---

## Running tests

```bash
pytest   # 235 tests across 4 files
```

---

## License

MIT
