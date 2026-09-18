"""Causal-uplift CLI: run CUPED variance reduction + uplift modeling.

Usage:
    uv run python run.py            # full pipeline
    uv run python run.py --no-plot
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from src.cuped import run_cuped
from src.uplift import run_uplift
from src import eval as ueval

REPORTS = pathlib.Path(__file__).parent / "reports"


def _print_cuped(cuped: dict) -> None:
    n = cuped["naive"]; c = cuped["cuped"]
    print("\n=== CUPED (variance reduction) ===")
    print(f"covariate=x_pre, outcome=y_cont, rho={cuped['rho']:.3f}, theta={cuped['theta']:.3f}")
    print(f"{'method':<8}{'ATE':>9}{'SE':>9}   {'CI95':<}")
    print(f"{'naive':<8}{n['ate']:>9.3f}{n['se']:>9.3f}   [{n['ci_low']:.3f}, {n['ci_high']:.3f}]")
    print(f"{'cuped':<8}{c['ate']:>9.3f}{c['se']:>9.3f}   [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    print(f"variance reduction: theory={cuped['var_reduction_theory']:.3f}  "
          f"actual={cuped['var_reduction_actual']:.3f}  "
          f"(SE x{cuped['se_ratio']:.3f}, CI narrowed x{cuped['ci_narrowing']:.2f})")


def _print_uplift(results: dict) -> None:
    print("\n=== Uplift modeling (test set) ===")
    print(f"{'model':<12}{'AUUC':>9}{'QINI':>9}{'uplift@20%':>12}{'corr(tau)':>11}")
    for name, r in results.items():
        it = r["ite_recovery"]
        print(f"{name:<12}{r['auuc']:>9.4f}{r['qini']:>9.4f}{r['uplift_at_top20']:>12.3f}"
              f"{it['corr_with_true']:>11.3f}")

    print("\n=== Segment-level uplift recovery ===")
    for name in ("t_learner", "s_learner"):
        print(f" {name}:")
        for seg, d in results[name]["segments"].items():
            print(f"   {seg:<11} pred={d['predicted']:.3f}  truth={d['ground_truth']:.2f}  err={d['abs_err']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(pathlib.Path(__file__).parent / "data" / "experiment.csv")
    cuped = run_cuped(df, y_col="y_cont", x_col="x_pre")
    _print_cuped(cuped)

    up = run_uplift(df)
    test = up["test"]
    y, w = test["y_bin"], test["treatment"]
    results = {}
    for name, u in (("t_learner", up["t_learner"]),
                    ("s_learner", up["s_learner"]),
                    ("random", up["random"])):
        results[name] = {
            "auuc": ueval.auuc(u, y, w),
            "qini": ueval.qini_coefficient(u, y, w),
            "uplift_at_top20": ueval.uplift_at_top_k(u, y, w, 0.20),
            "ite_recovery": ueval.ite_recovery(u, test),
            "segments": ueval.segment_uplift(u, test, df),
        }
    _print_uplift(results)

    out = {
        "cuped": {"rho": cuped["rho"], "theta": cuped["theta"],
                  "var_reduction_theory": cuped["var_reduction_theory"],
                  "var_reduction_actual": cuped["var_reduction_actual"],
                  "naive": cuped["naive"], "cuped": cuped["cuped"]},
        "uplift": results,
        "n": {"train": up["train_n"], "test": up["test_n"]},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))

    if not args.no_plot:
        try:
            _plot(up, test)
        except Exception as e:  # plotting optional in headless CI
            print(f"[plot skipped: {e}]")


def _plot(up, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = test["y_bin"].values
    w = test["treatment"].values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in (("t_learner", "C0"), ("s_learner", "C1"), ("random", "C2")):
        dfc = ueval._curve(up[name], pd.Series(y), pd.Series(w))
        frac = np.arange(1, len(dfc) + 1) / len(dfc)
        ax[0].plot(frac, dfc["qini"].values, label=name, color=color)
    ax[0].set_title("QINI / uplift curves")
    ax[0].set_xlabel("population sorted by predicted uplift (desc)")
    ax[0].set_ylabel("incremental responders")
    ax[0].legend()

    gt = {"new": 0.60, "returning": 0.08}
    segs = ("new", "returning")
    x = np.arange(2)
    ax[1].bar(x - 0.15, [up["t_learner"][test.segment == s].mean() for s in segs],
              0.3, label="T-learner")
    ax[1].bar(x + 0.15, [up["s_learner"][test.segment == s].mean() for s in segs],
              0.3, label="S-learner")
    ax[1].scatter(x, [gt[s] for s in segs], color="red", zorder=5,
                  marker="_", s=300, linewidths=3, label="ground truth")
    ax[1].set_xticks(x); ax[1].set_xticklabels(list(segs))
    ax[1].set_title("Segment-level uplift vs ground truth")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].legend()
    fig.tight_layout()
    out = REPORTS / "uplift.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[saved {out}]")


if __name__ == "__main__":
    main()
