"""Figures for causal-uplift EDA.

Produces 5 PNGs (dpi=300) under reports/eda/figures/:
  1. distribution of the primary metric y_cont (overall + by arm)
  2. conversion (y_bin) by segment x treatment  -> lead segment finding
  3. CUPED: naive vs CUPED ATE with 95% CI       -> variance-reduction finding
  4. covariate x_pre vs outcome y_cont (hexbin)  -> why CUPED works (rho=0.667)
  5. ground-truth tau_true vs observed diff-in-means by segment (scale damping)

Run:
  /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
      reports/eda/scripts/plot_figures.py
"""

from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")  # headless: before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
DATA = PROJECT / "data" / "experiment.csv"
FIGDIR = PROJECT / "reports" / "eda" / "figures"
RES = json.loads((HERE / "analysis_results.json").read_text())

FIGDIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)

COL_CTRL = "#4C72B0"
COL_TRT = "#DD8452"
COL_NEW = "#C44E52"
COL_RET = "#55A868"


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}")


# ---- Fig 1: distribution of y_cont ----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
y = df["y_cont"]
axs[0].hist(y, bins=60, color="#8172B3", edgecolor="white", linewidth=0.4)
axs[0].axvline(
    y.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean = {y.mean():.2f}"
)
axs[0].axvline(
    y.median(), color="#DD8452", linestyle=":", linewidth=1.5, label=f"median = {y.median():.2f}"
)
axs[0].set(
    xlabel="y_cont (continuous outcome)",
    ylabel="Users",
    title=f"y_cont distribution (skew = {RES['uni_y_cont']['skew']:.2f})",
)
axs[0].legend(frameon=False)

for w, c, lab in [(0, COL_CTRL, "control"), (1, COL_TRT, "treatment")]:
    axs[1].hist(
        df.loc[df.treatment == w, "y_cont"],
        bins=50,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color=c,
        label=lab,
    )
axs[1].set(
    xlabel="y_cont (continuous outcome)",
    ylabel="Density",
    title="y_cont by treatment arm (overlapping)",
)
axs[1].legend(frameon=False)
save(fig, "fig1_distribution_y_cont.png")

# ---- Fig 2: conversion by segment x treatment (lead finding) --------------
seg = RES["by_segment"]
segs = ["new", "returning"]
ctrl = [seg[s]["y_bin_control"] for s in segs]
trt = [seg[s]["y_bin_treatment"] for s in segs]
x = np.arange(len(segs))
w = 0.36
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
b1 = ax.bar(x - w / 2, ctrl, w, label="control", color=COL_CTRL, edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w / 2, trt, w, label="treatment", color=COL_TRT, edgecolor="black", linewidth=0.5)
for b in (b1, b2):
    ax.bar_label(b, fmt="%.3f", padding=2, fontsize=9)
for i, s in enumerate(segs):
    gap = seg[s]["y_bin_gap"]
    lo, hi = seg[s]["y_bin_gap_ci"]
    ax.annotate(
        f"gap = {gap:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]",
        xy=(i, max(ctrl[i], trt[i]) + 0.09),
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#B8860B", lw=0.8),
    )
ax.set_xticks(x)
ax.set_xticklabels(segs)
ax.set(
    ylim=(0, 1.08),
    xlabel="segment",
    ylabel="Conversion rate (y_bin)",
    title="Conversion uplift by segment: new responds ~8.4x returning",
)
ax.legend(frameon=False, loc="lower right")
save(fig, "fig2_conversion_by_segment.png")

# ---- Fig 3: CUPED naive vs adjusted ATE with CI ---------------------------
cup = RES["cuped"]
labels = ["Naive", "CUPED"]
ates = [cup["naive"]["ate"], cup["cuped"]["ate"]]
errs = [1.96 * cup["naive"]["se"], 1.96 * cup["cuped"]["se"]]
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
bars = ax.bar(
    labels,
    ates,
    yerr=errs,
    capsize=8,
    color=["#8C8C8C", "#4C72B0"],
    edgecolor="black",
    linewidth=0.6,
    width=0.55,
)
ax.bar_label(bars, fmt="ATE = %.3f", padding=18, fontsize=10)
ax.axhline(cup["naive"]["ate"], color="#8C8C8C", linestyle="--", linewidth=1)
ax.annotate(
    f"SE: {cup['naive']['se']:.4f} -> {cup['cuped']['se']:.4f}  "
    f"(-{cup['se_reduction_pct'] * 100:.1f}%)\n"
    f"variance reduction: {cup['var_reduction_actual']:.3f} actual "
    f"vs {cup['var_reduction_theory']:.3f} theory (rho^2)",
    xy=(0.5, 0.06),
    xycoords="axes fraction",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", fc="#E8F4EA", ec="#2E7D32", lw=0.8),
)
ax.set(
    ylim=(0.20, 0.36),
    xlabel="estimator",
    ylabel="ATE on y_cont (95% CI)",
    title="CUPED: same ATE, ~26% smaller standard error",
)
save(fig, "fig3_cuped_ci.png")

# ---- Fig 4: covariate vs outcome ------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
hb = ax.hexbin(df["x_pre"], df["y_cont"], gridsize=45, cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax, label="users per bin")
rho = RES["corr_x_pre_y_cont"]
ax.set(
    xlabel="x_pre (pre-period covariate)",
    ylabel="y_cont (continuous outcome)",
    title=f"Pre-period covariate vs outcome (Pearson r = {rho:.3f}) - CUPED input",
)
ax.text(
    0.02,
    0.97,
    f"rho^2 = {rho**2:.3f}  ->  variance reduction floor",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#666", lw=0.7),
)
save(fig, "fig4_covariate_vs_outcome.png")

# ---- Fig 5: ground truth vs observed, by segment --------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
tau = [seg[s]["tau_true"] for s in segs]
obs = [seg[s]["y_cont_ate"]["ate"] for s in segs]
obs_err = [1.96 * seg[s]["y_cont_ate"]["se"] for s in segs]
b1 = ax.bar(
    x - w / 2,
    tau,
    w,
    label="tau_true (latent ground truth)",
    color="#8172B3",
    edgecolor="black",
    linewidth=0.5,
)
b2 = ax.bar(
    x + w / 2,
    obs,
    w,
    yerr=obs_err,
    capsize=6,
    label="observed diff-in-means (y_cont, 95% CI)",
    color="#DD8452",
    edgecolor="black",
    linewidth=0.5,
)
ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
ax.bar_label(b2, fmt="%.3f", padding=6, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(segs)
ax.set(
    xlabel="segment",
    ylabel="Treatment effect on y_cont",
    title="Ground-truth tau vs observed effect: recovering tau by segment",
)
ax.legend(frameon=False, loc="upper right")
save(fig, "fig5_tau_vs_observed.png")

print("done")
