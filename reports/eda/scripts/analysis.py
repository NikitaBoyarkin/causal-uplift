"""EDA analysis for causal-uplift — Phase 1-4 computations (no Phase 5: no time column).

Recomputes every headline number the README claims, checks SRM, validity,
duplicates, correlations, leakage of `tau_true`, and segment-level effects.

Run:
  /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
      reports/eda/scripts/analysis.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
DATA = PROJECT / "data" / "experiment.csv"
OUT = HERE / "analysis_results.json"

EXPECTED = {
    "user_id",
    "segment",
    "x_pre",
    "treatment",
    "y_cont",
    "y_bin",
    "tau_true",
}


def q1_q3(s: pd.Series) -> tuple[float, float]:
    return float(s.quantile(0.25)), float(s.quantile(0.75))


def iqr_outlier_share(s: pd.Series, k: float = 1.5) -> float:
    q1, q3 = q1_q3(s)
    iqr = q3 - q1
    return float(((s < q1 - k * iqr) | (s > q3 + k * iqr)).mean())


def ate_with_se(df: pd.DataFrame, y_col: str) -> dict:
    t = df.loc[df.treatment == 1, y_col]
    c = df.loc[df.treatment == 0, y_col]
    ate = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - 1.96 * se,
        "ci_high": ate + 1.96 * se,
        "n_t": int(len(t)),
        "n_c": int(len(c)),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    res: dict = {}

    # ---- structure / grain -------------------------------------------------
    res["shape"] = list(df.shape)
    res["columns"] = list(df.columns)
    res["columns_match_expected"] = set(df.columns) == EXPECTED
    res["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    res["user_id_unique"] = bool(df["user_id"].is_unique)
    res["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    res["full_row_dups"] = int(df.duplicated().sum())
    res["key_dups"] = int(df.duplicated(subset=["user_id"]).sum())

    # ---- validity ----------------------------------------------------------
    res["treatment_values"] = sorted(df["treatment"].unique().tolist())
    res["segment_values"] = sorted(df["segment"].unique().tolist())
    res["y_bin_values"] = sorted(df["y_bin"].unique().tolist())
    res["tau_true_values"] = sorted(df["tau_true"].unique().tolist())
    res["x_pre_range"] = [float(df["x_pre"].min()), float(df["x_pre"].max())]
    res["y_cont_range"] = [float(df["y_cont"].min()), float(df["y_cont"].max())]
    res["y_bin_rate"] = float(df["y_bin"].mean())

    # ---- SRM ---------------------------------------------------------------
    n_t = int((df["treatment"] == 1).sum())
    n_c = int((df["treatment"] == 0).sum())
    n = len(df)
    chi2, p_srm = stats.chisquare([n_t, n_c], f_exp=[n / 2, n / 2])
    res["srm"] = {
        "n_treatment": n_t,
        "n_control": n_c,
        "share_treatment": n_t / n,
        "share_control": n_c / n,
        "expected": n / 2,
        "chi2": float(chi2),
        "p_value": float(p_srm),
        "verdict": "PASS (no SRM)" if p_srm >= 0.001 else "FAIL (SRM)",
    }
    # segment split vs design 30/70
    n_new = int((df["segment"] == "new").sum())
    chi2_seg, p_seg = stats.chisquare([n_new, n - n_new], f_exp=[0.30 * n, 0.70 * n])
    res["segment_split"] = {
        "n_new": n_new,
        "n_returning": n - n_new,
        "share_new": n_new / n,
        "share_returning": (n - n_new) / n,
        "chi2_vs_30_70": float(chi2_seg),
        "p_value": float(p_seg),
    }
    # segment x treatment balance (randomization check)
    ct = pd.crosstab(df["segment"], df["treatment"])
    chi2_ct, p_ct, dof, _ = stats.chi2_contingency(ct)
    res["segment_x_treatment"] = {
        "table": ct.to_dict(),
        "chi2": float(chi2_ct),
        "p_value": float(p_ct),
        "dof": int(dof),
    }
    # covariate balance on x_pre
    t_x = df.loc[df.treatment == 1, "x_pre"]
    c_x = df.loc[df.treatment == 0, "x_pre"]
    tt, p_x = stats.ttest_ind(t_x, c_x, equal_var=False)
    smd = (t_x.mean() - c_x.mean()) / np.sqrt((t_x.var(ddof=1) + c_x.var(ddof=1)) / 2)
    res["x_pre_balance"] = {
        "mean_t": float(t_x.mean()),
        "mean_c": float(c_x.mean()),
        "diff": float(t_x.mean() - c_x.mean()),
        "t": float(tt),
        "p_value": float(p_x),
        "smd": float(smd),
    }

    # ---- univariate --------------------------------------------------------
    for col in ["x_pre", "y_cont", "tau_true"]:
        s = df[col]
        res[f"uni_{col}"] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
            "skew": float(s.skew()),
            "iqr_outlier_share": iqr_outlier_share(s),
        }
    res["uni_y_bin"] = {"mean": float(df["y_bin"].mean()), "share_1": float(df["y_bin"].mean())}
    res["cardinality"] = {
        c: int(df[c].nunique()) for c in ["segment", "treatment", "y_bin", "tau_true"]
    }

    # ---- bivariate ---------------------------------------------------------
    num = ["x_pre", "y_cont", "y_bin", "treatment", "tau_true"]
    res["corr_pearson"] = df[num].corr(method="pearson").round(4).to_dict()
    res["corr_spearman"] = df[num].corr(method="spearman").round(4).to_dict()
    res["corr_x_pre_y_cont"] = float(df["x_pre"].corr(df["y_cont"]))

    # ---- leakage audit: tau_true vs features ------------------------------
    res["leakage"] = {
        "features_in_model": ["x_pre", "segment"],
        "tau_true_used_as_feature": False,
        "corr_tau_x_pre": float(df["tau_true"].corr(df["x_pre"])),
        "corr_tau_segment": float(df["tau_true"].corr((df["segment"] == "new").astype(int))),
        "note": (
            "tau_true is a deterministic function of `segment` "
            "(0.60 new / 0.08 returning); it is ground truth, not a feature. "
            "`segment` is a legitimate pre-treatment stratifier, so the model can "
            "recover tau_true only through the segment split — no leakage."
        ),
    }

    # ---- CUPED -------------------------------------------------------------
    var_x = np.var(df["x_pre"], ddof=1)
    cov = np.cov(df["y_cont"], df["x_pre"], ddof=1)[0, 1]
    theta = cov / var_x
    x_mean = df["x_pre"].mean()
    df["y_adj"] = df["y_cont"] - theta * (df["x_pre"] - x_mean)
    rho = float(np.corrcoef(df["y_cont"], df["x_pre"])[0, 1])
    naive = ate_with_se(df, "y_cont")
    cuped = ate_with_se(df, "y_adj")
    se_ratio = cuped["se"] / naive["se"]
    res["cuped"] = {
        "rho": rho,
        "theta": float(theta),
        "var_reduction_theory": float(rho**2),
        "var_reduction_actual": float(1 - (cuped["se"] ** 2) / (naive["se"] ** 2)),
        "se_ratio": float(se_ratio),
        "se_reduction_pct": float(1 - se_ratio),
        "ci_narrowing": float(
            (naive["ci_high"] - naive["ci_low"]) / (cuped["ci_high"] - cuped["ci_low"])
        ),
        "naive": naive,
        "cuped": cuped,
        "n_per_arm_for_same_power": float(10000 * se_ratio**2),
    }

    # ---- treatment effect overall + by segment ----------------------------
    res["ate_observed_y_cont"] = ate_with_se(df, "y_cont")
    res["ate_true_mean_tau"] = float(df["tau_true"].mean())

    seg_rows = {}
    for seg in ["new", "returning"]:
        f = df[df["segment"] == seg]
        # continuous diff-in-means
        cont = ate_with_se(f, "y_cont")
        # binary conversion gap with bootstrap CI
        t = f.loc[f.treatment == 1, "y_bin"].to_numpy(float)
        c = f.loc[f.treatment == 0, "y_bin"].to_numpy(float)
        rng = np.random.default_rng(42)
        draws = np.array(
            [
                t[rng.integers(0, len(t), len(t))].mean()
                - c[rng.integers(0, len(c), len(c))].mean()
                for _ in range(500)
            ]
        )
        seg_rows[seg] = {
            "n": int(len(f)),
            "n_t": int((f.treatment == 1).sum()),
            "n_c": int((f.treatment == 0).sum()),
            "tau_true": float(f["tau_true"].iloc[0]),
            "y_cont_ate": cont,
            "y_bin_control": float(c.mean()),
            "y_bin_treatment": float(t.mean()),
            "y_bin_gap": float(t.mean() - c.mean()),
            "y_bin_gap_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "x_pre_mean": float(f["x_pre"].mean()),
        }
    res["by_segment"] = seg_rows
    res["new_vs_returning_ratio"] = (
        seg_rows["new"]["y_bin_gap"] / seg_rows["returning"]["y_bin_gap"]
    )
    res["new_vs_returning_tau_ratio"] = (
        seg_rows["new"]["tau_true"] / seg_rows["returning"]["tau_true"]
    )

    # ---- metric by treatment arm ------------------------------------------
    res["by_treatment"] = {
        str(int(w)): {
            "n": int((df.treatment == w).sum()),
            "y_cont_mean": float(df.loc[df.treatment == w, "y_cont"].mean()),
            "y_bin_mean": float(df.loc[df.treatment == w, "y_bin"].mean()),
            "x_pre_mean": float(df.loc[df.treatment == w, "x_pre"].mean()),
        }
        for w in [0, 1]
    }

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    print(f"SRM: n_t={n_t} n_c={n_c} chi2={chi2:.3f} p={p_srm:.4f} -> {res['srm']['verdict']}")
    print(f"CUPED se_ratio={se_ratio:.4f} reduction={1 - se_ratio:.4f}")
    print(
        f"new gap={seg_rows['new']['y_bin_gap']:.4f} returning gap={seg_rows['returning']['y_bin_gap']:.4f} "
        f"ratio={res['new_vs_returning_ratio']:.2f}"
    )
    print(
        f"ATE y_cont observed={res['ate_observed_y_cont']['ate']:.4f} "
        f"tau_true mean={res['ate_true_mean_tau']:.4f}"
    )


if __name__ == "__main__":
    main()
