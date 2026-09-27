"""Uplift modeling: estimate the individual treatment effect (ITE).

Two simple, dependency-light learners built on LightGBM:

  T-learner: fit one model per treatment arm on (X | treatment), predict
             counterfactuals, uplift = p_t - p_c.
  S-learner: fit one model on (X, treatment) -> y, uplift = pred(t=1) - pred(t=0).

Evaluated against ground-truth `tau_true` (available in synthetic data) and via
AUUC / QINI / uplift@top-k on a held-out test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from . import config as C

FEATURES = ["x_pre", "segment"]

# One shared dtype for `segment` across every frame that reaches LightGBM.
# Deriving categories per-frame would give the same string different integer
# codes in different arms, and LightGBM consumes those codes as split values
# without complaining.
SEGMENT_DTYPE = pd.CategoricalDtype(["new", "returning"], ordered=False)


def _X(df: pd.DataFrame) -> pd.DataFrame:
    seg = df["segment"]
    unexpected = set(seg.unique()) - set(SEGMENT_DTYPE.categories)
    if unexpected:
        raise ValueError(
            f"unknown segment value(s) {sorted(unexpected)}; "
            f"expected one of {list(SEGMENT_DTYPE.categories)}"
        )
    return df[FEATURES].assign(segment=seg.astype(SEGMENT_DTYPE))


def t_learner(train: pd.DataFrame, test: pd.DataFrame, y_col: str = "y_bin") -> np.ndarray:
    c = train[train.treatment == 0]
    t = train[train.treatment == 1]
    mc = LGBMClassifier(**C.LGBM_PARAMS).fit(_X(c), c[y_col])
    mt = LGBMClassifier(**C.LGBM_PARAMS).fit(_X(t), t[y_col])
    return mt.predict_proba(_X(test))[:, 1] - mc.predict_proba(_X(test))[:, 1]


def s_learner(train: pd.DataFrame, test: pd.DataFrame, y_col: str = "y_bin") -> np.ndarray:
    X = _X(train).assign(treatment=train.treatment.values)
    m = LGBMClassifier(**C.LGBM_PARAMS).fit(X, train[y_col])
    Xc = _X(test).assign(treatment=0)
    Xt = _X(test).assign(treatment=1)
    return m.predict_proba(Xt)[:, 1] - m.predict_proba(Xc)[:, 1]


def random_uplift(test: pd.DataFrame, seed: int = C.SEED) -> np.ndarray:
    return np.random.default_rng(seed).random(len(test))


def run_uplift(df: pd.DataFrame, y_bin: str = "y_bin") -> dict:
    if y_bin not in df.columns:
        raise ValueError(f"outcome column {y_bin!r} not in frame: {list(df.columns)}")
    train, test = train_test_split(df, test_size=0.3, random_state=C.SEED, stratify=df.treatment)
    up_t = t_learner(train, test, y_bin)
    up_s = s_learner(train, test, y_bin)
    up_r = random_uplift(test)
    return {
        "train_n": len(train), "test_n": len(test),
        "t_learner": up_t, "s_learner": up_s, "random": up_r,
        "test": test.reset_index(drop=True),
    }
