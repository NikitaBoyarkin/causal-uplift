"""Config for causal-uplift: seed + LightGBM learner params."""
from __future__ import annotations

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
    "value_per_conversion": 40.0,    # money units per incremental conversion
    "campaign_cost_per_user": 2.0,   # money units per targeted user
    "ci_level": 0.95,
}
