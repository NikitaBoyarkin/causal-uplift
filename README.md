# Causal / Uplift

Causal inference for product experiments — two methods that go beyond
"two-sample t-test on the outcome":

1. **CUPED** — variance reduction using a pre-period covariate. Same ATE,
   smaller CI → experiments need fewer users to reach significance.
2. **Uplift modeling** — estimate the *individual* treatment effect (ITE),
   so retention/discount actions target users who actually respond, not
   everyone.

Both run on a synthetic randomized A/B experiment with a known, heterogeneous
treatment effect, so the estimates can be checked against ground truth.

## What it shows

### CUPED (variance reduction)

| Method | ATE | SE | 95% CI |
|---|---|---|---|
| Naive | 0.270 | 0.019 | [0.232, 0.308] |
| CUPED | 0.276 | 0.014 | [0.247, 0.304] |

The point estimate is unchanged (0.270 → 0.276, within noise). The standard
error shrinks by ~26% (×0.74), and the CI narrows 1.35×. Variance reduction
~45% actual vs 55% theoretical (the gap is the covariate being a pre-period
proxy, not the outcome itself). Concretely: an experiment that needed 10k
users per arm now needs ~5.6k for the same power.

### Uplift (ITE recovery)

| Model | AUUC | QINI | uplift@20% | corr(τ) |
|---|---|---|---|---|
| T-learner | 0.0043 | 0.0014 | 0.072 | 0.50 |
| S-learner | 0.0057 | 0.0029 | 0.041 | 0.66 |
| Random | 0.0014 | −0.0014 | −0.015 | 0.007 |

Both learners beat random on every metric. Segment-level recovery (predicted
vs empirical ground truth):

| Segment | Truth (binary uplift) | T-learner | S-learner |
|---|---|---|---|
| new | 0.110 | 0.126 | 0.120 |
| returning | 0.010 | 0.018 | 0.019 |

The model recovers that "new" users respond ~10× more than "returning" users —
the targeting signal a discount campaign would act on.

A note on honesty: the latent ground-truth τ is 0.60 (new) / 0.08 (returning),
but the *conversion* uplift is ~0.11 / 0.01 because the sigmoid at a high
baseline conversion (~71%) damps large latent effects. Comparing predicted
binary uplift to latent τ would be a scale mismatch; we report rank
correlation (scale-free) and per-segment empirical recovery (same scale).

## Data

Synthetic, deterministic (seed = 42). 20,000 users in a randomized experiment:

- `segment` — `new` (30%) / `returning` (70%) → heterogeneous treatment effect.
- `x_pre` — pre-period covariate, correlated with the outcome (ρ ≈ 0.67).
- `treatment` — 50/50, independent of everything (clean randomization).
- `tau_true` — the ground-truth individual treatment effect (eval only).
- `y_cont` — continuous outcome; `y_bin` — binary conversion.

## Methods

- **CUPED**: `Y_adj = Y − θ·(X − mean(X))`, `θ = Cov(Y,X)/Var(X)`. The ATE
  estimate on `Y_adj` has the same expectation and ~`(1 − ρ²)` of the variance.
- **T-learner**: one model per treatment arm; uplift = `P(t=1) − P(t=0)`.
- **S-learner**: one model with `treatment` as a feature;
  uplift = `pred(t=1) − pred(t=0)`.
- **Evaluation**: AUUC, Qini coefficient, uplift@top-20%, rank correlation
  with latent τ, and per-segment empirical recovery.

Implemented from scratch on LightGBM (no `causalml`/`econml` dependency) so the
mechanics are visible. In production these would be `econml`'s
`T Learner` / `X Learner` or `causalml`'s uplift forests.

## Quick start

```bash
uv run --with pandas --with numpy python data/generate_data.py
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib python run.py
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib --with pytest pytest -q
```

Outputs: `reports/metrics.json` + `reports/uplift.png` (QINI curves + segment
uplift vs ground truth).

## Layout

```
causal-uplift/
├── data/generate_data.py      # synthetic randomized experiment + HTE
├── src/
│   ├── config.py              # seed + LGBM learner params
│   ├── cuped.py               # CUPED adjustment + ATE/SE
│   ├── uplift.py              # T-learner / S-learner
│   └── eval.py                # AUUC, QINI, uplift@k, recovery
├── tests/                     # data sanity, CUPED, uplift-beats-random, recovery
├── reports/                   # metrics.json + uplift.png (gitignored)
├── conftest.py
└── run.py
```

## Notes

- CUPED buys power for free if you have a pre-period covariate — no new
  experiment design, just a better estimator on data you already collected.
- Uplift modeling answers a different question than A/B testing: not "does the
  treatment work on average" but "who does it work on." The two are
  complementary, not substitutes.
- The synthetic data has a true heterogeneous effect, which is the only reason
  recovery can be checked. On real data you never observe the ITE — that is the
  fundamental problem of causal inference.
