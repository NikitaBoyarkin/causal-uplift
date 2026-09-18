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
}
