"""Tests for the causal-uplift pipeline.

Run: uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib --with pytest pytest -q
"""
import numpy as np
import pandas as pd

from src.cuped import run_cuped
from src.uplift import run_uplift
from src import eval as ueval


def _df():
    return pd.read_csv("data/experiment.csv")


def test_data_sanity():
    df = _df()
    assert len(df) == 20000
    assert abs(df.treatment.mean() - 0.5) < 0.02          # randomized
    ate_obs = df[df.treatment == 1].y_cont.mean() - df[df.treatment == 0].y_cont.mean()
    ate_true = df.tau_true.mean()
    assert abs(ate_obs - ate_true) < 0.06                 # observed ATE close to truth
    assert df.x_pre.corr(df.y_cont) > 0.5                  # CUPED covariate is informative


def test_cuped_preserves_ate_and_shrinks_se():
    r = run_cuped(_df(), y_col="y_cont", x_col="x_pre")
    # ATE point estimate essentially unchanged
    assert abs(r["cuped"]["ate"] - r["naive"]["ate"]) < 0.02
    # Standard error strictly smaller
    assert r["cuped"]["se"] < r["naive"]["se"]
    assert r["var_reduction_actual"] > 0.30


def test_uplift_beats_random():
    up = run_uplift(_df())
    test = up["test"]
    y, w = test["y_bin"], test["treatment"]
    rand_corr = ueval.ite_recovery(up["random"], test)["corr_with_true"]
    rand_auuc = ueval.auuc(up["random"], y, w)
    for name in ("t_learner", "s_learner"):
        corr = ueval.ite_recovery(up[name], test)["corr_with_true"]
        auuc = ueval.auuc(up[name], y, w)
        assert corr > rand_corr + 0.2, f"{name} corr {corr} not clearly > random {rand_corr}"
        assert auuc > rand_auuc, f"{name} AUUC {auuc} not > random {rand_auuc}"


def test_segment_uplift_recovery():
    df = _df()
    up = run_uplift(df)
    test = up["test"]
    for name in ("t_learner", "s_learner"):
        seg = ueval.segment_uplift(up[name], test, df)
        for seg_name, d in seg.items():
            assert d["abs_err"] < 0.05, f"{name}/{seg_name} abs_err {d['abs_err']}"
