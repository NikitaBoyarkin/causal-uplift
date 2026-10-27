"""CUPED (Controlled-Pre-Experiment-Data): variance reduction for A/B tests.

Given a pre-period covariate X correlated with the outcome Y, the adjusted
outcome Y_adj = Y - theta*(X - mean(X)), theta = Cov(Y,X)/Var(X), has the same
expected treatment effect but smaller variance. The ATE point estimate is
unchanged up to theta*(mean(X|t) - mean(X|c)), which vanishes under covariate
balance; its standard error shrinks by ~sqrt(1 - rho^2).

Conventions used here:
  * `var_reduction_*` are the *reduction* (rho^2), not the retained fraction
    (1 - rho^2). They are the same quantity the theory predicts, so theory and
    actual are directly comparable.
  * `se_ratio = se_cuped / se_naive` is sqrt(1 - rho^2).

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # imported as part of the package
    from . import config as C
except ImportError:  # loaded as a standalone file by scripts/check_claims.py
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _spec = _ilu.spec_from_file_location("_causal_config", _Path(__file__).with_name("config.py"))
    C = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(C)


def cuped_adjust(df: pd.DataFrame, y_col: str, x_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return (df with a `y_adj` column, theta/rho diagnostics)."""
    var_x = np.var(df[x_col], ddof=1)
    if not np.isfinite(var_x) or var_x <= 0:
        raise ValueError(
            f"covariate {x_col!r} has non-positive variance ({var_x}) - CUPED is undefined"
        )
    cov = np.cov(df[y_col], df[x_col], ddof=1)[0, 1]
    theta = cov / var_x
    x_mean = df[x_col].mean()
    out = df.copy()
    out["y_adj"] = df[y_col] - theta * (df[x_col] - x_mean)
    rho = np.corrcoef(df[y_col], df[x_col])[0, 1]
    if not np.isfinite(rho):
        raise ValueError(f"correlation of {y_col!r} and {x_col!r} is not finite")
    return out, {
        "theta": float(theta),
        "rho": float(rho),
        # rho^2 is the variance *reduction*; 1 - rho^2 is the retained fraction.
        "var_reduction_theory": float(rho**2),
    }


def ate_with_se(df: pd.DataFrame, y_col: str, ci_level: float | None = None) -> dict:
    t = df.loc[df.treatment == 1, y_col]
    c = df.loc[df.treatment == 0, y_col]
    if len(t) < 2 or len(c) < 2:
        raise ValueError(f"need >=2 observations per arm, got treatment={len(t)} control={len(c)}")
    ate = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    z = C.z_multiplier(ci_level)
    ci = (ate - z * se, ate + z * se)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "var_t": t.var(ddof=1),
        "var_c": c.var(ddof=1),
        "n_t": len(t),
        "n_c": len(c),
    }


def run_cuped(
    df: pd.DataFrame, y_col: str = "y_cont", x_col: str = "x_pre", ci_level: float | None = None
) -> dict:
    naive = ate_with_se(df, y_col, ci_level=ci_level)
    adj_df, meta = cuped_adjust(df, y_col, x_col)
    cuped = ate_with_se(adj_df, "y_adj", ci_level=ci_level)
    actual_reduction = 1 - (cuped["se"] ** 2) / (naive["se"] ** 2)
    return {
        "covariate": x_col,
        "outcome": y_col,
        "theta": meta["theta"],
        "rho": meta["rho"],
        "var_reduction_theory": meta["var_reduction_theory"],
        "var_reduction_actual": actual_reduction,
        "naive": naive,
        "cuped": cuped,
        "se_ratio": cuped["se"] / naive["se"],
        "ci_narrowing": (naive["ci_high"] - naive["ci_low"]) / (cuped["ci_high"] - cuped["ci_low"]),
    }
