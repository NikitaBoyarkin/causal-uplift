"""Config for causal-uplift: seed + LightGBM learner params + CI level."""

from __future__ import annotations

from statistics import NormalDist

SEED = 42

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": SEED,
    # Cross-platform reproducibility: without these, histogram sums depend on the
    # thread count and the committed reference stops matching on CI.
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
}

# Pre-registered ship / no_ship / inconclusive rule (see docs/preregistration.md).
DECISION = {
    "target_segment": "new",
    "value_per_conversion": 40.0,  # money units per incremental conversion
    "campaign_cost_per_user": 2.0,  # money units per targeted user
    "ci_level": 0.95,
}


def resolve_ci_level(ci_level: float | None) -> float:
    """The configured level unless the caller overrides it; validated either way.

    ``None`` means "use ``DECISION['ci_level']``", so every interval-producing
    function defaults to the single configured knob instead of a local literal.
    """
    level = DECISION["ci_level"] if ci_level is None else float(ci_level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {level!r}")
    return level


def ci_percentiles(ci_level: float | None = None) -> tuple[float, float]:
    """Two-sided percentile bounds for a percentile bootstrap, in percent.

    0.95 -> (2.5, 97.5); 0.80 -> (10.0, 90.0).
    """
    alpha = 1.0 - resolve_ci_level(ci_level)
    return 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)


def z_multiplier(ci_level: float | None = None) -> float:
    """Two-sided normal z for the level: 0.95 -> 1.95996, 0.80 -> 1.28155."""
    return NormalDist().inv_cdf(0.5 + resolve_ci_level(ci_level) / 2.0)
