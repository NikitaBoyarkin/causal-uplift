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

The point estimate is unchanged within noise (0.270 → 0.276, shift 0.0053 —
exactly the covariate imbalance `θ(x̄t − x̄c)`, which a perfectly balanced split
would zero out). The standard error shrinks by ~26% (×0.742) and the CI narrows
1.35×.

Variance reduction: **0.449 actual vs 0.445 theory (ρ²)** — the two agree, as
they should: `x_pre` is a genuine pre-period covariate and the estimator is
unbiased. Concretely, an experiment that needed 10k users per arm now needs
**~5.5k** for the same power (10k × 0.742²).

### Uplift (ITE recovery)

| Model | AUUC | z vs null | QINI | z vs null | uplift@20% | corr(τ) | rank corr |
|---|---|---|---|---|---|---|---|
| `oracle_tau` (true τ) | 0.0066 | 2.25 | 0.0038 | 2.25 | 0.0558 | 1.000 | 1.000 |
| S-learner | 0.0057 | 1.72 | 0.0029 | 1.72 | 0.0405 | 0.657 | 0.621 |
| T-learner | 0.0043 | 0.81 | 0.0014 | 0.81 | 0.0724 | 0.495 | 0.458 |
| Random | 0.0077 | 2.90 | 0.0048 | 2.90 | 0.0590 | 0.011 | 0.011 |

Null reference (200 random scorers on the same test split): AUUC
+0.00293 ± 0.00164, QINI +0.00009 ± 0.00164.

The honest reading is **not** "both learners beat random". At n = 6000 the curve
metrics are noise-dominated, and the table says so three ways:

- The T-learner sits at **0.81σ** — indistinguishable from random.
- The S-learner is at **1.72σ** — marginal at best.
- The **oracle**, which ranks users by the *known true* individual effect, is
  only at **2.25σ**. That is the ceiling; nothing above it is achievable here.
- The single seeded random scorer happens to land at **2.90σ**, above both real
  learners. That is what a ±0.0016 null looks like when you draw from it once.

Every 95% bootstrap interval straddles zero and overlaps every other model:

| Model | AUUC 95% CI | QINI 95% CI | uplift@20% 95% CI |
|---|---|---|---|
| `oracle_tau` | [−0.0015, 0.0135] | [0.0003, 0.0067] | [0.0025, 0.1117] |
| S-learner | [−0.0006, 0.0127] | [−0.0005, 0.0061] | [−0.0211, 0.0990] |
| T-learner | [−0.0026, 0.0114] | [−0.0017, 0.0050] | [0.0187, 0.1311] |
| Random | [0.0014, 0.0139] | [0.0015, 0.0081] | [0.0063, 0.1068] |

What *is* statistically clear at this sample size is the **ranking quality** and
the **segment split**, both of which have a reference that is not noise:

| Segment | Empirical gap (95% CI) | T-learner | S-learner | n (test) |
|---|---|---|---|---|
| `new` | 0.1108 [0.0894, 0.1357] | 0.126 | 0.120 | 1782 |
| `returning` | 0.0132 [−0.0013, 0.0271] | 0.018 | 0.019 | 4218 |

`corr(τ)` is 0.50–0.66 against 0.011 for random, and the model recovers that
"new" users respond ~8× more than "returning" users — the targeting signal a
discount campaign would act on. Note the reference CI: the predicted means
(a 6k test split) and the reference (the full 20k) carry separate sampling
error, so `abs_err ≈ 0.015` means little until the intervals are compared —
they overlap comfortably.

Two footnotes on honesty:

- The latent ground-truth τ is 0.60 (`new`) / 0.08 (`returning`), but the
  *conversion* uplift is ~0.11 / 0.01 because the sigmoid at a high baseline
  conversion (~71%) damps large latent effects. Comparing predicted binary
  uplift to latent τ would be a scale mismatch; the binary comparison above is
  on the same scale.
- `tau_true` takes only **two** values (one per segment), so `corr(τ)` is a
  between-segment point-biserial correlation, not a within-segment ITE ranking.
  Both Pearson and rank versions are reported; they differ only in tie handling.

## Data

Synthetic, deterministic (seed = 42). 20,000 users in a randomized experiment:

- `segment` — `new` (30%) / `returning` (70%) → heterogeneous treatment effect.
- `x_pre` — pre-period covariate, correlated with the outcome (ρ ≈ 0.67).
- `treatment` — 50/50, independent of everything (clean randomization).
- `tau_true` — the ground-truth individual treatment effect (eval only).
- `y_cont` — continuous outcome; `y_bin` — binary conversion.

## Methods

- **CUPED**: `Y_adj = Y − θ·(X − mean(X))`, `θ = Cov(Y,X)/Var(X)`. The ATE
  estimate on `Y_adj` has the same expectation and ~`(1 − ρ²)` of the variance,
  i.e. a variance *reduction* of `ρ²`.
- **T-learner**: one model per treatment arm; uplift = `P(t=1) − P(t=0)`.
- **S-learner**: one model with `treatment` as a feature;
  uplift = `pred(t=1) − pred(t=0)`.
- **Evaluation**: AUUC, Qini coefficient, uplift@top-20%, correlation with
  latent τ, and per-segment empirical recovery — each paired with a reference.

### Reading a curve metric honestly

Three helpers in `src/eval.py` keep the curve metrics from being read as
evidence on their own:

- `metric_null(y, w, fn)` — the metric's distribution under *random* scoring
  (200 draws, seeded). Note AUUC is **not** centered at zero: because the curve
  uses a running `cum_t/cum_c` ratio, a random scorer's AUUC sits near `C/(2n)`,
  so a raw AUUC carries a positive offset.
- `bootstrap_ci(fn, uplift, y, w)` — percentile interval on the test split.
- `z_vs_null(value, null)` — how many null SDs above random the value sits.

`run.py` prints all three per model, plus an `oracle_tau` row (ranking by the
known τ) as the achievable ceiling. It also writes `metrics.json` with
`allow_nan=False`, so an undefined metric fails loudly instead of emitting
invalid JSON.

Implemented from scratch on LightGBM (no `causalml`/`econml` dependency) so the
mechanics are visible. In production these would be `econml`'s
`T Learner` / `X Learner` or `causalml`'s uplift forests.

## Quick start

```bash
uv sync
uv run python data/generate_data.py   # writes data/experiment.csv
uv run python run.py                  # prints tables, writes reports/
uv run pytest -q                      # 23 tests
uv run ruff check .                   # lint
```

Options: `--n-boot` (bootstrap resamples, 400) and `--n-null` (null draws, 200)
trade runtime for tighter references.

Outputs: `reports/metrics.json` + `reports/uplift.png` (QINI curves + segment
uplift against the empirical gap, with its CI).

## Layout

```
causal-uplift/
├── data/generate_data.py      # synthetic randomized experiment + HTE
├── src/
│   ├── config.py              # seed + LGBM learner params
│   ├── cuped.py               # CUPED adjustment + ATE/SE
│   ├── uplift.py              # T-learner / S-learner
│   └── eval.py                # AUUC, QINI, uplift@k, recovery, null/CI
├── tests/                     # 23 tests incl. mutation-verified guards
├── reports/                   # metrics.json + uplift.png (generated, gitignored)
├── run.py                     # CLI
├── uv.lock                    # committed: reproducible environment
└── pyproject.toml
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
- The most transferable lesson here is the sample-size one: with ~6k test rows
  and a ~30% baseline conversion, AUUC/QINI cannot separate a *perfect* ranker
  from noise. Reporting a bare QINI without a null reference would have made the
  T-learner look like a real result.
