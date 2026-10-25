"""Deterministic synthetic experiment data for causal-uplift.

A randomized A/B experiment with a pre-period covariate (for CUPED) and a
heterogeneous treatment effect by segment (for uplift modeling).

Reproducible: seed=42. Run: uv run python data/generate_data.py
Import: `from data.generate_data import build_dataset, write_dataset`
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
SEED = 42
N = 20_000


def build_dataset(seed: int = SEED, n: int = N) -> pd.DataFrame:
    """Build the synthetic experiment frame; the same seed gives the same frame."""
    rng = np.random.default_rng(seed)
    # Segment: new users get a LARGE treatment effect; returning users small.
    is_new = rng.random(n) < 0.30
    segment = np.where(is_new, "new", "returning")

    # Pre-period covariate (correlated with the outcome baseline).
    x_pre = np.where(is_new, rng.normal(0.0, 1.0, n), rng.normal(1.5, 1.0, n))

    # Baseline continuous outcome under no treatment.
    baseline = 2.0 + 0.7 * x_pre + np.where(is_new, -0.3, 0.3)

    # Randomized assignment (independent of X_pre / segment).
    w = (rng.random(n) < 0.5).astype(int)

    # Heterogeneous treatment effect (the ground-truth uplift).
    tau = np.where(is_new, 0.60, 0.08)

    noise = rng.normal(0, 1.0, n)
    y_cont = baseline + w * tau + noise  # continuous outcome

    # Binary conversion via sigmoid of the continuous latent.
    p_conv = 1 / (1 + np.exp(-(y_cont - 1.6)))
    y_bin = (rng.random(n) < p_conv).astype(int)

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "segment": segment,
            "x_pre": np.round(x_pre, 4),
            "treatment": w,
            "y_cont": np.round(y_cont, 4),
            "y_bin": y_bin,
            "tau_true": tau,  # ground-truth individual treatment effect (eval only)
        }
    )


def write_dataset(path: str | pathlib.Path, seed: int = SEED, n: int = N) -> pd.DataFrame:
    """Write CSV to path (creating parent dirs) and return the frame."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(seed=seed, n=n)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    """Keep current behaviour + printed summary; write HERE/'experiment.csv'."""
    df = write_dataset(HERE / "experiment.csv")
    print(f"Generated {len(df):,} rows (seed={SEED})")
    print(f"  control conversion: {df[df.treatment == 0].y_bin.mean():.3f}")
    print(f"  treatment conversion: {df[df.treatment == 1].y_bin.mean():.3f}")
    print(f"  ATE (true): {df.tau_true.mean():.3f}")
    print(
        f"  ATE (observed): {df[df.treatment == 1].y_cont.mean() - df[df.treatment == 0].y_cont.mean():.3f}"
    )
    print(f"  corr(x_pre, y_cont): {df.x_pre.corr(df.y_cont):.3f}")


if __name__ == "__main__":
    main()
