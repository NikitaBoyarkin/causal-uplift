"""Uplift evaluation: AUUC, QINI, uplift@top-k, and (with ground truth) ITE recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd

_trap = getattr(np, "trapezoid", None) or np.trapz


def _curve(uplift, y, w):
    """Sort by predicted uplift desc; cumulative incremental responders."""
    df = pd.DataFrame({"u": uplift, "y": y.values, "w": w.values}).sort_values("u", ascending=False)
    df["cum_t"] = (df.w == 1).cumsum()
    df["cum_c"] = (df.w == 0).cumsum()
    df["cum_y_t"] = ((df.w == 1) & (df.y == 1)).cumsum()
    df["cum_y_c"] = ((df.w == 0) & (df.y == 1)).cumsum()
    # QINI-style incremental: cum_y_t - cum_y_c * (cum_t/cum_c)
    df["qini"] = df["cum_y_t"] - df["cum_y_c"] * (df["cum_t"] / df["cum_c"].clip(lower=1))
    df["qini"] = df["qini"].fillna(0)
    return df


def auuc(uplift, y, w) -> float:
    """Area under the uplift curve (mean incremental gain per population fraction)."""
    df = _curve(uplift, y, w)
    n = len(df)
    frac = np.arange(1, n + 1) / n
    # AUUC = integral of qini / n (normalize by population)
    return float(_trap(df["qini"].values, frac) / n)


def qini_coefficient(uplift, y, w) -> float:
    """Qini coefficient = AUUC(model) - AUUC(random assignment)."""
    df = _curve(uplift, y, w)
    n = len(df)
    frac = np.arange(1, n + 1) / n
    # Random reference: total incremental gain scaled by fraction.
    total_t = (w == 1).sum(); total_c = (w == 0).sum()
    yt = ((w == 1) & (y == 1)).sum(); yc = ((w == 0) & (y == 1)).sum()
    random_qini = (yt - yc * total_t / max(1, total_c)) * frac
    return float(_trap(df["qini"].values - random_qini, frac) / n)


def uplift_at_top_k(uplift, y, w, k_frac: float = 0.20) -> float:
    """Average uplift (treatment - control conversion) in the top-k by predicted uplift."""
    n_top = max(1, int(len(y) * k_frac))
    idx = np.argsort(uplift)[::-1][:n_top]
    ytop, wtop = y.values[idx], w.values[idx]
    t = (wtop == 1); c = (wtop == 0)
    if t.sum() == 0 or c.sum() == 0:
        return 0.0
    return float(ytop[t].mean() - ytop[c].mean())


def ite_recovery(uplift, test: pd.DataFrame) -> dict:
    """Ranking recovery vs the latent ground-truth tau.

    We only report rank correlation: the predicted uplift is on the conversion
    (probability) scale while tau_true is on the continuous-latent scale, so an
    absolute error would mix scales. Correlation is scale-free and tests whether
    the model orders users by treatment effect correctly.
    """
    tau_true = test["tau_true"].values
    u = np.asarray(uplift)
    return {"corr_with_true": float(np.corrcoef(u, tau_true)[0, 1])}


def segment_uplift(uplift, test: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Mean predicted uplift per segment vs the empirical binary ground truth.

    The latent continuous tau (0.60 / 0.08) is NOT comparable to predicted
    *conversion* uplift — the sigmoid damps it. Ground truth is the actual
    treatment-control conversion gap per segment, computed on the full dataset.
    """
    s = pd.Series(uplift, index=test.index)
    out = {}
    for seg in ("new", "returning"):
        f = full[full.segment == seg]
        gt = float(f[f.treatment == 1].y_bin.mean() - f[f.treatment == 0].y_bin.mean())
        pred = float(s[test.segment == seg].mean())
        out[seg] = {"predicted": round(pred, 4), "ground_truth": round(gt, 4),
                    "abs_err": round(abs(pred - gt), 4)}
    return out
