"""Tests for the eval primitives the reported intervals rest on.

`bootstrap_ci` produced every interval in the README with no test of its own;
these bind determinism, the bracket property, and error propagation, plus a
regression guard for the random-baseline / null-distribution collision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from data.generate_data import build_dataset
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


def test_ci_level_controls_bootstrap_interval_width():
    """The configured knob must change the interval, not just be echoed.

    0.80 and 0.95 use the same bootstrap draws (same seed), so only the
    percentiles differ: the 80% interval must sit strictly inside the 95% one.
    """
    uplift, y, w = _sample()
    narrow = ueval.bootstrap_ci(
        ueval.qini_coefficient, uplift, y, w, n_boot=200, seed=1, ci_level=0.80
    )
    wide = ueval.bootstrap_ci(
        ueval.qini_coefficient, uplift, y, w, n_boot=200, seed=1, ci_level=0.95
    )
    assert (narrow["hi"] - narrow["lo"]) < (wide["hi"] - wide["lo"])
    assert narrow["lo"] > wide["lo"]
    assert narrow["hi"] < wide["hi"]
    default = ueval.bootstrap_ci(ueval.qini_coefficient, uplift, y, w, n_boot=200, seed=1)
    assert default == wide, "the default interval must use DECISION['ci_level'] (0.95)"


def test_segment_truth_ci_level_changes_bounds():
    rng = np.random.default_rng(0)
    n = 600
    ref = pd.DataFrame(
        {
            "segment": ["new"] * n,
            "treatment": rng.integers(0, 2, n),
            "y_bin": (rng.random(n) < 0.2).astype(int),
        }
    )
    narrow = ueval.segment_truth(ref, n_boot=200, ci_level=0.80)["new"]
    wide = ueval.segment_truth(ref, n_boot=200, ci_level=0.95)["new"]
    assert (narrow["hi"] - narrow["lo"]) < (wide["hi"] - wide["lo"])
    assert narrow["lo"] > wide["lo"]
    assert narrow["hi"] < wide["hi"]


def test_ci_level_rejects_impossible_levels():
    with pytest.raises(ValueError, match="ci_level"):
        ueval.bootstrap_ci(ueval.qini_coefficient, *_sample(), n_boot=5, ci_level=1.5)
    with pytest.raises(ValueError, match="ci_level"):
        C.ci_percentiles(0.0)


def test_curve_tie_order_is_the_input_order():
    """Tied scores must be consumed in input order, not in sort-engine order.

    The oracle score takes two values, so almost every test row is tied; under a
    bare ``sort_values`` the curve order (and AUUC) depended on the algorithm.
    """
    rng = np.random.default_rng(11)
    n = 6000
    u = np.where(rng.random(n) < 0.5, 0.60, 0.08)
    y = (rng.random(n) < 0.30).astype(float)
    w = rng.integers(0, 2, n)
    expected = np.lexsort((np.arange(n), -u))
    curve = ueval._curve(u, y, w)
    assert np.array_equal(curve["u"].to_numpy(), u[expected])
    assert np.array_equal(curve["y"].to_numpy(), y[expected])


def test_uplift_at_top_k_breaks_ties_by_input_position():
    """With a constant score the top-k must be the first k rows, deterministically.

    The old ``argsort(u)[::-1]`` reversed an all-tied block and picked the LAST
    rows, so the metric depended on the sort implementation.
    """
    n = 100
    u = np.full(n, 1.0)
    w = np.tile([1, 0], n // 2)
    y = np.zeros(n)
    y[: n // 5] = w[: n // 5]  # conversions only among treated rows of the first 20
    assert ueval.uplift_at_top_k(u, y, w, 0.20) == pytest.approx(1.0)


def test_oracle_metrics_are_pinned_to_the_tie_broken_order():
    """Pin the reproducible oracle row: the B118 regression target.

    Before the tiebreak the same data produced AUUC 0.006612 / QINI 0.003777 /
    uplift@20% 0.055785; the tie-broken (input-order) values are below.
    """
    df = build_dataset()
    _, test = train_test_split(df, test_size=0.3, random_state=C.SEED, stratify=df.treatment)
    test = test.reset_index(drop=True)
    u = test["tau_true"].to_numpy()
    y = test["y_bin"].to_numpy()
    w = test["treatment"].to_numpy()
    assert ueval.auuc(u, y, w) == pytest.approx(0.006084004122827816, abs=1e-9)
    assert ueval.qini_coefficient(u, y, w) == pytest.approx(0.00324902070322754, abs=1e-9)
    assert ueval.uplift_at_top_k(u, y, w, 0.20) == pytest.approx(0.04625829913588897, abs=1e-9)


def test_random_baseline_is_not_the_first_null_draw():
    """Shared seeds made the "random" row null draw #1 -- the 2.90-sigma artifact.

    That draw is the maximum of the 200 nulls on this data, so random outranked
    the oracle and broke the oracle-is-the-ceiling claim; the seeds must differ.
    """
    test = pd.DataFrame({"i": np.arange(6000)})
    first_null_draw = np.random.default_rng(C.SEED).random(len(test))
    assert not np.array_equal(random_uplift(test), first_null_draw)
