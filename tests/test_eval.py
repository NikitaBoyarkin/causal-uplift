"""Tests for the eval primitives the reported intervals rest on.

`bootstrap_ci` produced every interval in the README with no test of its own;
these bind determinism, the bracket property, and error propagation, plus a
regression guard for the random-baseline / null-distribution collision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import eval as ueval
from src.uplift import random_uplift


def _sample(n: int = 2000, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A well-behaved slice: real treatment/control gap, score ranks it."""
    rng = np.random.default_rng(seed)
    uplift = rng.normal(size=n)
    w = rng.integers(0, 2, n)
    tau = 0.02 + 0.10 * (uplift > 0)
    p = np.clip(0.10 + tau * w, 0.0, 1.0)
    y = (rng.random(n) < p).astype(float)
    return uplift, y, w


def _ci(seed: int, n_boot: int = 100) -> dict:
    uplift, y, w = _sample()
    return ueval.bootstrap_ci(ueval.qini_coefficient, uplift, y, w, n_boot=n_boot, seed=seed)


def test_bootstrap_ci_is_deterministic_in_its_seed():
    assert _ci(seed=1) == _ci(seed=1)
    assert _ci(seed=1) != _ci(seed=2)


def test_bootstrap_ci_brackets_the_full_sample_estimate():
    uplift, y, w = _sample()
    full = ueval.qini_coefficient(uplift, y, w)
    ci = ueval.bootstrap_ci(ueval.qini_coefficient, uplift, y, w, n_boot=100, seed=1)
    assert ci["lo"] < ci["hi"]
    assert ci["lo"] < full < ci["hi"]
    assert ci["sd"] > 0


def test_bootstrap_ci_propagates_single_arm_error():
    """A single-arm slice makes the metric undefined; the error must not be swallowed."""
    n = 200
    y = np.zeros(n)
    w = np.zeros(n)  # all control: qini_coefficient raises
    with pytest.raises(ValueError, match="both arms"):
        ueval.bootstrap_ci(ueval.qini_coefficient, np.arange(n, dtype=float), y, w, n_boot=20)


def test_random_baseline_is_not_the_first_null_draw():
    """Shared seeds made the "random" row null draw #1 -- the 2.90-sigma artifact.

    That draw is the maximum of the 200 nulls on this data, so random outranked
    the oracle and broke the oracle-is-the-ceiling claim; the seeds must differ.
    """
    test = pd.DataFrame({"i": np.arange(6000)})
    first_null_draw = np.random.default_rng(C.SEED).random(len(test))
    assert not np.array_equal(random_uplift(test), first_null_draw)
