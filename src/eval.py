"""Uplift evaluation: AUUC, QINI, uplift@top-k, ITE recovery, and null/CI context.

Every curve metric is a *sample* statistic on the held-out test split, and at
the sample sizes here they are noise-dominated: even a perfect ranking of the
known `tau_true` lands only ~2 sigma above a random scorer on QINI. So each
metric is reported next to an empirical null distribution (`metric_null`) and a
bootstrap interval (`bootstrap_ci`), and `run.py` prints the z-score against the
null. A bare QINI with no reference is not evidence of anything.

All functions accept numpy arrays or pandas Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

_trap = getattr(np, "trapezoid", None) or np.trapz


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties (e.g. tau_true taking two values) are handled."""
    return pd.Series(np.asarray(x)).rank().to_numpy()


def _curve(uplift, y, w) -> pd.DataFrame:
    """Sort by predicted uplift desc; cumulative incremental responders."""
    df = pd.DataFrame({
        "u": np.asarray(uplift, dtype=float),
        "y": np.asarray(y, dtype=float),
        "w": np.asarray(w, dtype=float),
    }).sort_values("u", ascending=False)
    df["cum_t"] = (df.w == 1).cumsum()
    df["cum_c"] = (df.w == 0).cumsum()
    df["cum_y_t"] = ((df.w == 1) & (df.y == 1)).cumsum()
    df["cum_y_c"] = ((df.w == 0) & (df.y == 1)).cumsum()
    # QINI-style incremental: cum_y_t - cum_y_c * (cum_t/cum_c)
    df["qini"] = df["cum_y_t"] - df["cum_y_c"] * (df["cum_t"] / df["cum_c"].clip(lower=1))
    return df


def auuc(uplift, y, w) -> float:
    """Area under the uplift curve, normalized by population (`/n`).

    Convention: mean incremental responders per person, so the value is bounded
    by the ATE and is not comparable to AUUC conventions that skip the `/n`.
    """
    df = _curve(uplift, y, w)
    n = len(df)
    frac = np.arange(1, n + 1) / n
    return float(_trap(df["qini"].values, frac) / n)


def qini_coefficient(uplift, y, w) -> float:
    """Qini coefficient = AUUC(model) - AUUC(random assignment), exactly."""
    total_t = int((np.asarray(w) == 1).sum())
    total_c = int((np.asarray(w) == 0).sum())
    if total_t == 0 or total_c == 0:
        raise ValueError(
            f"qini is undefined without both arms, got treatment={total_t} control={total_c}"
        )
    df = _curve(uplift, y, w)
    n = len(df)
    frac = np.arange(1, n + 1) / n
    yt = ((df.w == 1) & (df.y == 1)).sum()
    yc = ((df.w == 0) & (df.y == 1)).sum()
    random_qini = (yt - yc * total_t / total_c) * frac
    return float(_trap(df["qini"].values - random_qini, frac) / n)


def uplift_at_top_k(uplift, y, w, k_frac: float = 0.20) -> float:
    """Mean treatment-control conversion gap in the top-k by predicted uplift.

    Unbiased for the top-k subset ATE: selection depends on the score only, and
    the arm split inside the subset stays ~50/50 (measured 51.0/49.0 at k=0.2).
    Returns NaN when the subset is single-arm - "cannot compute" must not look
    like "no effect".
    """
    u = np.asarray(uplift)
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(w)
    n_top = max(1, int(len(yv) * k_frac))
    idx = np.argsort(u)[::-1][:n_top]
    ytop, wtop = yv[idx], wv[idx]
    t, c = (wtop == 1), (wtop == 0)
    if t.sum() == 0 or c.sum() == 0:
        return float("nan")
    return float(ytop[t].mean() - ytop[c].mean())


def ite_recovery(uplift, test: pd.DataFrame) -> dict:
    """Agreement with the latent ground-truth tau.

    `tau_true` takes only two values (one per segment), so both statistics are
    *between-segment*: a point-biserial correlation, not a within-segment ITE
    ranking. Both are reported; only the rank version matches the "rank
    correlation" wording used in the README.
    """
    tau = np.asarray(test["tau_true"], dtype=float)
    u = np.asarray(uplift, dtype=float)
    nan = float("nan")
    if len(tau) < 2 or np.unique(tau).size < 2 or np.std(u) == 0:
        return {"corr_pearson_with_true": nan, "rank_corr_with_true": nan}
    return {
        "corr_pearson_with_true": float(np.corrcoef(u, tau)[0, 1]),
        "rank_corr_with_true": float(np.corrcoef(_rank(u), _rank(tau))[0, 1]),
    }


def segment_truth(ref: pd.DataFrame, segments=None, n_boot: int = 200,
                  seed: int = C.SEED) -> dict:
    """Empirical per-segment conversion gap (treatment - control) with a CI.

    This is the reference the predicted *binary* uplift is compared against -
    NOT the latent `tau_true`, which lives on the continuous scale and is damped
    by the sigmoid (comparing the two is a scale mismatch). Computed on the full
    frame by default: it is a marginal segment ATE, not a model prediction, and
    the extra rows make it a tighter estimate, not a leaked one.
    """
    if segments is None:
        segments = tuple(sorted(ref["segment"].unique()))
    segments = tuple(segments)
    rng = np.random.default_rng(seed)
    n = len(ref)
    out = {}
    for seg in segments:
        f = ref[ref.segment == seg]
        t = f.loc[f.treatment == 1, "y_bin"].to_numpy(dtype=float)
        c = f.loc[f.treatment == 0, "y_bin"].to_numpy(dtype=float)
        if len(t) == 0 or len(c) == 0:
            raise ValueError(f"segment {seg!r} has an empty arm - check the segment levels")
        draws = np.empty(n_boot)
        for i in range(n_boot):
            ti = t[rng.integers(0, len(t), len(t))]
            ci_ = c[rng.integers(0, len(c), len(c))]
            draws[i] = ti.mean() - ci_.mean()
        out[seg] = {
            "value": float(t.mean() - c.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "n_t": len(t), "n_c": len(c), "n_total": n,
        }
    return out


def segment_uplift(uplift, test: pd.DataFrame, truth: dict, n_boot: int = 200,
                   seed: int = C.SEED) -> dict:
    """Mean predicted uplift per segment vs the empirical gap from `segment_truth`.

    The reference CI is reported alongside, because the predicted mean (6k test
    rows) and the reference (20k rows) carry separate sampling error: an
    `abs_err` of 0.015 means little unless the intervals are shown.
    """
    u = np.asarray(uplift, dtype=float)
    s = pd.Series(u, index=test.index)
    rng = np.random.default_rng(seed)
    out = {}
    for seg, ref in truth.items():
        vals = s[test.segment == seg].to_numpy()
        if vals.size == 0:
            raise ValueError(f"segment {seg!r} absent from the test split")
        draws = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                          for _ in range(n_boot)])
        pred = float(vals.mean())
        out[seg] = {
            "predicted": round(pred, 4),
            "predicted_ci": [round(float(np.percentile(draws, 2.5)), 4),
                             round(float(np.percentile(draws, 97.5)), 4)],
            "ground_truth": round(ref["value"], 4),
            "ground_truth_ci": [round(ref["lo"], 4), round(ref["hi"], 4)],
            "abs_err": round(abs(pred - ref["value"]), 4),
            "n_test": int(vals.size),
        }
    return out


def metric_null(y, w, fn, n_draws: int = 200, seed: int = C.SEED) -> dict:
    """Null distribution of a curve metric under random scoring.

    Deterministic given `seed`. Note AUUC is *not* centered at zero here: with
    the running `cum_t/cum_c` ratio the random reference sits near C/(2n), so a
    raw AUUC carries a positive offset and must be read against this null.
    """
    yv, wv = np.asarray(y), np.asarray(w)
    rng = np.random.default_rng(seed)
    n = len(yv)
    vals = np.array([fn(rng.random(n), yv, wv) for _ in range(n_draws)], dtype=float)
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "q025": float(np.percentile(vals, 2.5)),
        "q975": float(np.percentile(vals, 97.5)),
        "n_draws": n_draws,
    }


def bootstrap_ci(fn, uplift, y, w, n_boot: int = 400, seed: int = C.SEED) -> dict:
    """Percentile bootstrap interval for a curve metric on (uplift, y, w)."""
    u = np.asarray(uplift, dtype=float)
    yv, wv = np.asarray(y), np.asarray(w)
    rng = np.random.default_rng(seed)
    n = len(yv)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = fn(u[idx], yv[idx], wv[idx])
    return {
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "sd": float(vals.std(ddof=1)),
    }


def z_vs_null(value: float, null: dict) -> float:
    """How many null standard deviations the observed value sits above random."""
    if null["sd"] == 0 or not np.isfinite(value):
        return float("nan")
    return float((value - null["mean"]) / null["sd"])
