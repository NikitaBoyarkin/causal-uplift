# EDA Report

**Rows:** 20000 · **Columns:** 7 · **Memory:** 1.21 MB

## Profile

| Column | dtype |
|---|---|
| user_id | int64 |
| segment | str |
| x_pre | float64 |
| treatment | int64 |
| y_cont | float64 |
| y_bin | int64 |
| tau_true | float64 |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| user_id | 20000 | 10000.5 | 10000.5 | 5773.647 | 5000.75 | 15000.25 | 19000.05 | 0.0% | 0.0 |
| x_pre | 20000 | 1.0435 | 1.0828 | 1.209 | 0.2362 | 1.8904 | 2.9565 | 0.5% | -0.1649 |
| treatment | 20000 | 0.495 | 0.0 | 0.5 | 0.0 | 1.0 | 1.0 | 0.0% | 0.0198 |
| y_cont | 20000 | 2.9545 | 2.9915 | 1.3807 | 2.0277 | 3.8972 | 5.1732 | 0.6% | -0.1103 |
| y_bin | 20000 | 0.7342 | 1.0 | 0.4418 | 0.0 | 1.0 | 1.0 | 0.0% | -1.0604 |
| tau_true | 20000 | 0.2364 | 0.08 | 0.2385 | 0.08 | 0.6 | 0.6 | 0.0% | 0.869 |

## Categorical Slices

**segment** — кардинальность 2

| Value | Count | Share |
|---|---:|---:|
| returning | 13985 | 69.9% |
| new | 6015 | 30.1% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| x_pre ~ y_cont | 0.6667 | moderate |
| x_pre ~ tau_true | -0.5657 | moderate |

## Metric by Segment

### segment (overall mean = 2.9545, gap x1.6757)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| returning | 13985 | 3.3622 | 47020.75 | 79.6% | +13.8% |
| new | 6015 | 2.0064 | 12068.5 | 20.4% | -32.1% |

### treatment (overall mean = 2.9545, gap x1.0959)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| 1.0 | 9901 | 3.091 | 30603.84 | 51.8% | +4.6% |
| 0.0 | 10099 | 2.8206 | 28485.42 | 48.2% | -4.5% |


## Findings

- **[INFO]** Разрыв по segment: x1.6757 между сегментами — кандидат в главный вывод

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
