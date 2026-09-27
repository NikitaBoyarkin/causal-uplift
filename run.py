"""Causal-uplift CLI: run CUPED variance reduction + uplift modeling.

Usage:
    uv run python run.py            # full pipeline
    uv run python run.py --no-plot

Every uplift metric is printed with a z-score against an empirical null
(random scorers) and a bootstrap interval, plus an `oracle_tau` row that ranks
users by the *true* individual effect - the ceiling any learner can reach.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from src import eval as ueval
from src.cuped import run_cuped
from src.uplift import run_uplift

ROOT = pathlib.Path(__file__).parent
REPORTS = ROOT / "reports"

MODELS = ("oracle_tau", "s_learner", "t_learner", "random")
CURVE_METRICS = ("auuc", "qini", "uplift_at_top20")


def _print_cuped(cuped: dict) -> None:
    n = cuped["naive"]
    c = cuped["cuped"]
    print("\n=== CUPED (variance reduction) ===")
    print(f"covariate=x_pre, outcome=y_cont, rho={cuped['rho']:.3f}, theta={cuped['theta']:.3f}")
    print(f"{'method':<8}{'ATE':>9}{'SE':>9}   CI95")
    print(f"{'naive':<8}{n['ate']:>9.3f}{n['se']:>9.3f}   [{n['ci_low']:.3f}, {n['ci_high']:.3f}]")
    print(f"{'cuped':<8}{c['ate']:>9.3f}{c['se']:>9.3f}   [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    print(f"variance reduction: theory(rho^2)={cuped['var_reduction_theory']:.3f}  "
          f"actual={cuped['var_reduction_actual']:.3f}  "
          f"(SE x{cuped['se_ratio']:.3f}, CI narrowed x{cuped['ci_narrowing']:.2f})")


def _print_uplift(results: dict, nulls: dict) -> None:
    print("\n=== Uplift modeling (held-out test set) ===")
    print(f"null ({nulls['qini']['n_draws']} random scorers): "
          f"AUUC {nulls['auuc']['mean']:+.5f} +/- {nulls['auuc']['sd']:.5f}   "
          f"QINI {nulls['qini']['mean']:+.5f} +/- {nulls['qini']['sd']:.5f}")
    print("z = (value - null mean) / null sd; z < ~2 means indistinguishable from random.\n")

    print(f"{'model':<11}{'AUUC':>8}{'z':>7}{'QINI':>9}{'z':>7}{'uplift@20%':>12}{'corr(tau)':>11}{'rank_corr':>11}")
    for name in MODELS:
        r = results[name]
        print(f"{name:<11}{r['auuc']:>8.4f}{r['auuc_z']:>7.2f}{r['qini']:>9.4f}"
              f"{r['qini_z']:>7.2f}{r['uplift_at_top20']:>12.4f}"
              f"{r['ite_recovery']['corr_pearson_with_true']:>11.3f}"
              f"{r['ite_recovery']['rank_corr_with_true']:>11.3f}")

    print("\n95% bootstrap intervals:")
    print(f"{'model':<11}{'AUUC':>19}{'QINI':>19}{'uplift@20%':>19}")
    for name in MODELS:
        r = results[name]
        cells = "".join(
            f"{r[f'{m}_ci']['lo']:>9.4f},{r[f'{m}_ci']['hi']:>8.4f}"
            for m in CURVE_METRICS
        )
        print(f"{name:<11}{cells}")

    print("\n=== Segment-level recovery (predicted binary uplift vs empirical gap) ===")
    for name in ("t_learner", "s_learner"):
        print(f" {name}:")
        for seg, d in results[name]["segments"].items():
            print(f"   {seg:<11} pred={d['predicted']:.3f} {d['predicted_ci']}  "
                  f"truth={d['ground_truth']:.4f} {d['ground_truth_ci']}  "
                  f"err={d['abs_err']:.3f} (n_test={d['n_test']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--n-boot", type=int, default=400, help="bootstrap resamples per metric")
    ap.add_argument("--n-null", type=int, default=200, help="random scorers for the null")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "experiment.csv")
    cuped = run_cuped(df, y_col="y_cont", x_col="x_pre")
    _print_cuped(cuped)

    up = run_uplift(df)
    test = up["test"]
    y, w = test["y_bin"], test["treatment"]

    segments = tuple(sorted(df["segment"].unique()))
    truth = ueval.segment_truth(df, segments)
    # Score every model against the SAME reference: the known tau_true.
    up["oracle_tau"] = test["tau_true"].to_numpy()

    metrics = {"auuc": ueval.auuc, "qini": ueval.qini_coefficient,
               "uplift_at_top20": lambda u, yy, ww: ueval.uplift_at_top_k(u, yy, ww, 0.20)}
    nulls = {name: ueval.metric_null(y, w, fn, n_draws=args.n_null) for name, fn in metrics.items()}

    results = {}
    for name in MODELS:
        u = up[name]
        row = {}
        for mname, fn in metrics.items():
            value = float(fn(u, y, w))
            row[mname] = value
            row[f"{mname}_ci"] = ueval.bootstrap_ci(fn, u, y, w, n_boot=args.n_boot)
            row[f"{mname}_z"] = ueval.z_vs_null(value, nulls[mname])
        row["ite_recovery"] = ueval.ite_recovery(u, test)
        row["segments"] = ueval.segment_uplift(u, test, truth)
        results[name] = row
    _print_uplift(results, nulls)

    out = {
        "cuped": {"rho": cuped["rho"], "theta": cuped["theta"],
                  "var_reduction_theory": cuped["var_reduction_theory"],
                  "var_reduction_actual": cuped["var_reduction_actual"],
                  "naive": cuped["naive"], "cuped": cuped["cuped"]},
        "uplift": results,
        "null": nulls,
        "segments": {k: {a: b for a, b in v.items() if a != "n_boot"} for k, v in truth.items()},
        "n": {"train": up["train_n"], "test": up["test_n"], "bootstrap": args.n_boot},
    }
    # allow_nan=False turns a silent NaN into a loud error - a NaN in metrics.json
    # would be invalid JSON and a quietly wrong README table.
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False))

    if not args.no_plot:
        _plot(up, test, truth)


def _plot(up, test, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = test["y_bin"].to_numpy()
    w = test["treatment"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in (("oracle_tau", "C3"), ("s_learner", "C1"),
                        ("t_learner", "C0"), ("random", "C2")):
        dfc = ueval._curve(up[name], y, w)
        frac = np.arange(1, len(dfc) + 1) / len(dfc)
        ax[0].plot(frac, dfc["qini"].values, label=name, color=color)
    ax[0].set_title("QINI / uplift curves")
    ax[0].set_xlabel("population sorted by predicted uplift (desc)")
    ax[0].set_ylabel("incremental responders")
    ax[0].legend()

    segs = tuple(truth)
    x = np.arange(len(segs))
    ax[1].bar(x - 0.15, [up["t_learner"][test.segment == s].mean() for s in segs],
              0.3, label="T-learner")
    ax[1].bar(x + 0.15, [up["s_learner"][test.segment == s].mean() for s in segs],
              0.3, label="S-learner")
    ax[1].errorbar(x - 0.15, [truth[s]["value"] for s in segs],
                   yerr=[[truth[s]["value"] - truth[s]["lo"] for s in segs],
                         [truth[s]["hi"] - truth[s]["value"] for s in segs]],
                   fmt="_", color="black", markersize=24, linewidth=2, zorder=5,
                   label="empirical gap (95% CI)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(list(segs))
    ax[1].set_title("Segment uplift vs empirical conversion gap")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend()
    fig.tight_layout()
    out = REPORTS / "uplift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    if not out.exists():
        raise RuntimeError(f"plot reported success but {out} is missing")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
