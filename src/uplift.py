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


def _X(df):
    return df[FEATURES].assign(segment=df["segment"].astype("category"))


def t_learner(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    c = train[train.treatment == 0]
    t = train[train.treatment == 1]
    mc = LGBMClassifier(**C.LGBM_PARAMS).fit(_X(c), c.y_bin)
    mt = LGBMClassifier(**C.LGBM_PARAMS).fit(_X(t), t.y_bin)
    return mt.predict_proba(_X(test))[:, 1] - mc.predict_proba(_X(test))[:, 1]


def s_learner(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    X = _X(train).assign(treatment=train.treatment.values)
    m = LGBMClassifier(**C.LGBM_PARAMS).fit(X, train.y_bin)
    Xc = _X(test).assign(treatment=0)
    Xt = _X(test).assign(treatment=1)
    return m.predict_proba(Xt)[:, 1] - m.predict_proba(Xc)[:, 1]


def random_uplift(test: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random(len(test))


def run_uplift(df: pd.DataFrame, y_bin: str = "y_bin") -> dict:
    train, test = train_test_split(df, test_size=0.3, random_state=C.SEED, stratify=df.treatment)
    up_t = t_learner(train, test)
    up_s = s_learner(train, test)
    up_r = random_uplift(test)
    return {
        "train_n": len(train), "test_n": len(test),
        "t_learner": up_t, "s_learner": up_s, "random": up_r,
        "test": test.reset_index(drop=True),
    }
