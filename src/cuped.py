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


def ate_with_se(df: pd.DataFrame, y_col: str) -> dict:
    t = df.loc[df.treatment == 1, y_col]
    c = df.loc[df.treatment == 0, y_col]
    if len(t) < 2 or len(c) < 2:
        raise ValueError(
            f"need >=2 observations per arm, got treatment={len(t)} control={len(c)}"
        )
    ate = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    return {"ate": ate, "se": se, "ci_low": ci[0], "ci_high": ci[1],
            "var_t": t.var(ddof=1), "var_c": c.var(ddof=1),
            "n_t": len(t), "n_c": len(c)}


def run_cuped(df: pd.DataFrame, y_col: str = "y_cont", x_col: str = "x_pre") -> dict:
    naive = ate_with_se(df, y_col)
    adj_df, meta = cuped_adjust(df, y_col, x_col)
    cuped = ate_with_se(adj_df, "y_adj")
    actual_reduction = 1 - (cuped["se"] ** 2) / (naive["se"] ** 2)
    return {
        "covariate": x_col, "outcome": y_col,
        "theta": meta["theta"], "rho": meta["rho"],
        "var_reduction_theory": meta["var_reduction_theory"],
        "var_reduction_actual": actual_reduction,
        "naive": naive,
        "cuped": cuped,
        "se_ratio": cuped["se"] / naive["se"],
        "ci_narrowing": (naive["ci_high"] - naive["ci_low"]) /
                        (cuped["ci_high"] - cuped["ci_low"]),
    }
