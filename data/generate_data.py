"""Deterministic synthetic experiment data for causal-uplift.

A randomized A/B experiment with a pre-period covariate (for CUPED) and a
heterogeneous treatment effect by segment (for uplift modeling).

Reproducible: seed=42. Run: uv run python data/generate_data.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
SEED = 42
N = 20_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    # Segment: new users get a LARGE treatment effect; returning users small.
    is_new = rng.random(N) < 0.30
    segment = np.where(is_new, "new", "returning")

    # Pre-period covariate (correlated with the outcome baseline).
    x_pre = np.where(is_new, rng.normal(0.0, 1.0, N), rng.normal(1.5, 1.0, N))

    # Baseline continuous outcome under no treatment.
    baseline = 2.0 + 0.7 * x_pre + np.where(is_new, -0.3, 0.3)

    # Randomized assignment (independent of X_pre / segment).
    w = (rng.random(N) < 0.5).astype(int)

    # Heterogeneous treatment effect (the ground-truth uplift).
    tau = np.where(is_new, 0.60, 0.08)

    noise = rng.normal(0, 1.0, N)
    y_cont = baseline + w * tau + noise  # continuous outcome

    # Binary conversion via sigmoid of the continuous latent.
    p_conv = 1 / (1 + np.exp(-(y_cont - 1.6)))
    y_bin = (rng.random(N) < p_conv).astype(int)

    df = pd.DataFrame({
        "user_id": np.arange(1, N + 1),
        "segment": segment,
        "x_pre": np.round(x_pre, 4),
        "treatment": w,
        "y_cont": np.round(y_cont, 4),
        "y_bin": y_bin,
        "tau_true": tau,  # ground-truth individual treatment effect (eval only)
    })
    df.to_csv(HERE / "experiment.csv", index=False)
    print(f"Generated {len(df):,} rows (seed={SEED})")
    print(f"  control conversion: {df[df.treatment==0].y_bin.mean():.3f}")
    print(f"  treatment conversion: {df[df.treatment==1].y_bin.mean():.3f}")
    print(f"  ATE (true): {tau.mean():.3f}")
    print(f"  ATE (observed): {df[df.treatment==1].y_cont.mean() - df[df.treatment==0].y_cont.mean():.3f}")
    print(f"  corr(x_pre, y_cont): {df.x_pre.corr(df.y_cont):.3f}")


if __name__ == "__main__":
    main()
