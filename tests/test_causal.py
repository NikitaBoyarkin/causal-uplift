"""Tests for the causal-uplift pipeline.

Run: uv sync && uv run pytest -q

Tests are split in two speeds: CUPED/eval tests are pure numerical checks (fast),
model tests share a module-scoped fixture so LightGBM trains once, and one
end-to-end test drives the CLI in a temporary working directory.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import eval as ueval
from src.cuped import ate_with_se, cuped_adjust, run_cuped
from src.uplift import _X, SEGMENT_DTYPE, run_uplift, t_learner

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data" / "experiment.csv"


def _df() -> pd.DataFrame:
    """Absolute path: tests must pass regardless of the invoking directory."""
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def up() -> dict:
    """Model results, trained once for the whole module."""
    return run_uplift(_df())


# --- data contract -----------------------------------------------------------

def test_data_sanity():
    df = _df()
    assert len(df) == 20000
    assert abs(df.treatment.mean() - 0.5) < 0.02          # randomized
    ate_obs = df[df.treatment == 1].y_cont.mean() - df[df.treatment == 0].y_cont.mean()
    ate_true = df.tau_true.mean()
    assert abs(ate_obs - ate_true) < 0.06                 # observed ATE close to truth
    assert df.x_pre.corr(df.y_cont) > 0.5                  # CUPED covariate is informative


def test_treatment_is_independent_of_segment_and_covariate():
    """A valid RCT: assignment must not depend on the segment or the covariate."""
    df = _df()
    for seg in sorted(df.segment.unique()):
        share = df.loc[df.segment == seg, "treatment"].mean()
        assert abs(share - 0.5) < 0.02, f"segment {seg} treated share {share}"
    gap = abs(df.loc[df.treatment == 1, "x_pre"].mean()
              - df.loc[df.treatment == 0, "x_pre"].mean())
    assert gap < 0.05, f"covariate imbalance {gap}"


# --- CUPED -------------------------------------------------------------------

def test_cuped_preserves_ate_and_shrinks_se():
    r = run_cuped(_df(), y_col="y_cont", x_col="x_pre")
    # ATE point estimate essentially unchanged under covariate balance
    assert abs(r["cuped"]["ate"] - r["naive"]["ate"]) < 0.02
    assert r["cuped"]["se"] < r["naive"]["se"]
    assert r["var_reduction_actual"] > 0.30


def test_cuped_theory_matches_actual():
    """`var_reduction_theory` must be the *reduction* (rho^2), so the two agree.

    Regression guard: the shipped value was `1 - rho**2` - the retained fraction -
    which printed a phantom 10pp "gap" against `var_reduction_actual`.
    """
    r = run_cuped(_df(), y_col="y_cont", x_col="x_pre")
    assert abs(r["var_reduction_theory"] - r["rho"] ** 2) < 1e-9
    assert abs(r["var_reduction_theory"] - r["var_reduction_actual"]) < 0.05


def test_cuped_shift_invariance():
    """Shifting the covariate must not change theta or the adjusted outcome."""
    df = _df()
    base, meta = cuped_adjust(df, "y_cont", "x_pre")
    shifted, meta_shift = cuped_adjust(df.assign(x_shift=df.x_pre + 1000), "y_cont", "x_shift")
    assert meta["theta"] == pytest.approx(meta_shift["theta"], rel=1e-9)
    assert np.allclose(base["y_adj"], shifted["y_adj"])


def test_cuped_uninformative_covariate_leaves_precision_unchanged():
    df = _df()
    df = df.assign(noise=np.arange(len(df)) % 7)
    r = run_cuped(df, y_col="y_cont", x_col="noise")
    assert abs(r["se_ratio"] - 1) < 0.05


def test_cuped_rejects_degenerate_covariate():
    df = _df().assign(const=1.0)
    with pytest.raises(ValueError, match="non-positive variance"):
        cuped_adjust(df, "y_cont", "const")


# --- eval primitives ---------------------------------------------------------

def test_ate_with_se_known_values():
    df = pd.DataFrame({"treatment": [1, 1, 1, 0, 0, 0], "y": [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]})
    r = ate_with_se(df, "y")
    assert r["ate"] == pytest.approx(3.0)
    assert r["var_t"] == pytest.approx(1.0)
    assert r["var_c"] == pytest.approx(1.0)
    assert r["se"] == pytest.approx(np.sqrt(1 / 3 + 1 / 3))
    assert r["ci_low"] < r["ate"] < r["ci_high"]


def test_ate_with_se_rejects_empty_arm():
    df = pd.DataFrame({"treatment": [1, 1], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="per arm"):
        ate_with_se(df, "y")


def test_uplift_at_top_k_finds_top_not_bottom():
    """Dropping the `[::-1]` (i.e. ranking the worst users first) must be caught.

    `x` is independent of the arm, so the top-k subset stays mixed; only the
    direction of the sort decides the sign of the gap.
    """
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    w = rng.integers(0, 2, size=n)
    p = 1 / (1 + np.exp(-(0.2 * x + 1.5 * x * w)))    # effect grows with x
    y = (rng.random(n) < p).astype(float)
    assert ueval.uplift_at_top_k(x, y, w, 0.2) > 0.15
    assert ueval.uplift_at_top_k(-x, y, w, 0.2) < -0.15
    # a constant score carries no signal at all
    assert ueval.uplift_at_top_k(np.full(n, 0.5), y, w, 0.2) == pytest.approx(0.0, abs=0.05)


def test_uplift_at_top_k_single_arm_is_nan_not_zero():
    """'Cannot compute' must not be reported as 'no effect'."""
    n = 100
    w = np.array([1] * 30 + [0] * 70)
    y = np.zeros(n)
    assert np.isnan(ueval.uplift_at_top_k(w.astype(float), y, w, 0.20))


def test_qini_rejects_single_arm():
    n = 10
    y = np.zeros(n)
    with pytest.raises(ValueError, match="both arms"):
        ueval.qini_coefficient(np.arange(n, dtype=float), y, np.ones(n))


def test_oracle_ranks_true_tau_perfectly_but_only_marginally_beats_noise(up):
    """The ceiling, and how low it is.

    Ranking by the known `tau_true` recovers it perfectly, yet at n=6000 its
    QINI lands only ~2 null-sd above random - so no learner's QINI here is
    evidence of anything. This test binds *both* halves of that claim.
    """
    test = up["test"]
    y, w = test["y_bin"], test["treatment"]
    oracle = test["tau_true"].to_numpy()
    assert ueval.ite_recovery(oracle, test)["corr_pearson_with_true"] == pytest.approx(1.0)

    null = ueval.metric_null(y, w, ueval.qini_coefficient, n_draws=80)
    z = ueval.z_vs_null(ueval.qini_coefficient(oracle, y, w), null)
    assert z > 1.5, f"oracle qini is not above random (z={z:.2f})"
    assert z < 5.0, f"oracle qini is supposed to be noise-dominated, got z={z:.2f}"


def test_qini_random_is_within_noise_of_zero(up):
    test = up["test"]
    y, w = test["y_bin"], test["treatment"]
    null = ueval.metric_null(y, w, ueval.qini_coefficient, n_draws=80)
    assert abs(null["mean"]) < 3 * null["sd"] / np.sqrt(null["n_draws"])
    assert abs(ueval.qini_coefficient(up["random"], y, w) - null["mean"]) < 3 * null["sd"]


def test_auuc_random_carries_a_positive_offset(up):
    """AUUC's random reference is not zero (running cum_t/cum_c ratio).

    So a raw AUUC must be read against this null, and the null mean should track
    the analytic linear reference C/(2n).
    """
    test = up["test"]
    y, w = test["y_bin"].to_numpy(), test["treatment"].to_numpy()
    null = ueval.metric_null(y, w, ueval.auuc, n_draws=80)
    yt = ((w == 1) & (y == 1)).sum()
    yc = ((w == 0) & (y == 1)).sum()
    analytic = (yt - yc * (w == 1).sum() / (w == 0).sum()) / (2 * len(y))
    assert null["mean"] > 0
    assert abs(null["mean"] - analytic) < 3 * null["sd"] / np.sqrt(null["n_draws"])


def test_ite_recovery_is_nan_when_tau_has_no_variance(up):
    test = up["test"].copy()
    test["tau_true"] = 0.5
    r = ueval.ite_recovery(up["s_learner"], test)
    assert np.isnan(r["corr_pearson_with_true"])
    assert np.isnan(r["rank_corr_with_true"])


# --- uplift models -----------------------------------------------------------

def test_segment_dtype_is_shared_across_arms():
    """Same string must map to the same code even if a subset lacks a level.

    Deriving the category per-frame gave `returning` code 0 in a subset without
    `new` but code 1 in the full frame; LightGBM consumes those codes silently.
    """
    df = _df()
    full = _X(df)
    only_returning = _X(df[df.segment == "returning"])
    assert full.segment.dtype == SEGMENT_DTYPE
    assert only_returning.segment.dtype == SEGMENT_DTYPE
    assert (only_returning.segment.cat.codes == 1).all()
    assert (full.loc[df.segment == "returning", "segment"].cat.codes == 1).all()


def test_unknown_segment_fails_loudly():
    df = _df().head(20).copy()
    df.loc[df.index[0], "segment"] = "vip"
    with pytest.raises(ValueError, match="unknown segment"):
        _X(df)


def test_uplift_beats_random(up):
    test = up["test"]
    _y, _w = test["y_bin"], test["treatment"]
    rand_corr = ueval.ite_recovery(up["random"], test)["corr_pearson_with_true"]
    for name in ("t_learner", "s_learner"):
        corr = ueval.ite_recovery(up[name], test)["corr_pearson_with_true"]
        assert corr > rand_corr + 0.2, f"{name} corr {corr} not clearly > random {rand_corr}"


def test_segment_uplift_recovery(up):
    df = _df()
    truth = ueval.segment_truth(df)
    assert set(truth) == {"new", "returning"}
    for name in ("t_learner", "s_learner"):
        seg = ueval.segment_uplift(up[name], up["test"], truth, n_boot=50)
        for seg_name, d in seg.items():
            assert d["abs_err"] < 0.05, f"{name}/{seg_name} abs_err {d['abs_err']}"
            # the reference must be the empirical conversion gap, not latent tau
            assert d["ground_truth"] < 0.5
            assert d["ground_truth_ci"][0] <= d["ground_truth"] <= d["ground_truth_ci"][1]


def test_segment_truth_rejects_unknown_segment():
    with pytest.raises(ValueError, match="empty arm"):
        ueval.segment_truth(_df(), segments=("vip",))


def test_learners_honour_the_requested_outcome_column():
    """`y_col` was accepted but ignored - the body read `.y_bin` directly."""
    df = _df().iloc[:900].assign(y_alt=lambda d: (d.x_pre > 0).astype(int))
    train, test = df.iloc[:600], df.iloc[600:]
    assert not np.allclose(t_learner(train, test, y_col="y_bin"),
                           t_learner(train, test, y_col="y_alt"))


# --- end to end --------------------------------------------------------------

def test_cli_writes_valid_json_and_plot(tmp_path):
    """Drives run.py from a foreign cwd; json.loads rejects a bare NaN.

    Artifacts are removed first: asserting they merely *exist* would pass off a
    stale file left by an earlier run.
    """
    for name in ("metrics.json", "uplift.png"):
        (ROOT / "reports" / name).unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--n-boot", "20", "--n-null", "20"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert set(out) >= {"cuped", "uplift", "null", "n"}
    assert set(out["uplift"]) >= {"oracle_tau", "t_learner", "s_learner", "random"}
    for row in out["uplift"].values():
        for key in ("auuc", "qini", "uplift_at_top20"):
            assert np.isfinite(row[key])
    assert (ROOT / "reports" / "uplift.png").exists()
