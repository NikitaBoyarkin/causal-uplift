"""Pre-registered decision rule: ``ship`` / ``no_ship`` / ``inconclusive``.

The verdict is driven only by the empirical per-segment conversion gap and its
bootstrap CI from ``eval.segment_truth`` - never by model-predicted uplift.
Curve metrics (AUUC / QINI / uplift@k) are noise-dominated at this sample size
and therefore have no vote (see ``docs/preregistration.md``).

The rule is cost-aware: a campaign is worth running only when the segment
uplift clears the break-even level set by campaign economics. The functions are
pure - no I/O, no RNG, no pandas.
"""

from __future__ import annotations

from . import config as C

SHIP = "ship"
NO_SHIP = "no_ship"
INCONCLUSIVE = "inconclusive"


def break_even_uplift(value_per_conversion: float, campaign_cost_per_user: float) -> float:
    """Uplift at which campaign revenue exactly covers campaign cost."""
    if value_per_conversion <= 0:
        raise ValueError(f"value_per_conversion must be > 0, got {value_per_conversion!r}")
    if campaign_cost_per_user < 0:
        raise ValueError(f"campaign_cost_per_user must be >= 0, got {campaign_cost_per_user!r}")
    return campaign_cost_per_user / value_per_conversion


def _target_size(seg: dict) -> int:
    """Users the campaign reaches: both arms of the target segment.

    The per-arm counts are required: ``n_total`` in ``eval.segment_truth`` is the
    whole frame (20 000 for every segment), so falling back to it would overstate
    the campaign size fivefold.
    """
    n_t = int(seg.get("n_t") or 0)
    n_c = int(seg.get("n_c") or 0)
    if n_t + n_c <= 0:
        raise ValueError(
            f"segment carries no per-arm counts (n_t={n_t}, n_c={n_c}); "
            "n_total is the whole frame and cannot size the campaign"
        )
    return n_t + n_c


def ship_decision(segments: dict, decision_cfg: dict | None = None) -> dict:
    """Apply the pre-registered rule to empirical segment results.

    ``segments`` is the output of ``eval.segment_truth`` (or the same shape):
    ``{segment: {"value", "lo", "hi", "n_t", "n_c", "n_total"}}``.

    Returns the decision plus the break-even uplift, the observed uplift and
    its CI, the targeted sample size, the expected net per user, and the cost
    of each error branch. Raises ``ValueError`` when the target segment is
    missing.
    """
    decision_cfg = C.DECISION if decision_cfg is None else decision_cfg
    target = decision_cfg["target_segment"]
    value = float(decision_cfg["value_per_conversion"])
    cost = float(decision_cfg["campaign_cost_per_user"])
    ci_level = float(decision_cfg.get("ci_level", 0.95))

    if target not in segments:
        raise ValueError(f"target segment {target!r} not present in segments")

    seg = segments[target]
    observed = float(seg["value"])
    lo = float(seg["lo"])
    hi = float(seg["hi"])
    n_target = _target_size(seg)

    be = break_even_uplift(value, cost)
    expected_net_per_user = observed * value - cost

    if lo > be:
        decision = SHIP
        rationale = (
            f"CI lower bound {lo:.4f} exceeds break-even {be:.4f}: "
            "even the pessimistic case pays for itself."
        )
    elif hi < be:
        decision = NO_SHIP
        rationale = (
            f"CI upper bound {hi:.4f} stays below break-even {be:.4f}: "
            "even the optimistic case loses money."
        )
    else:
        decision = INCONCLUSIVE
        rationale = (
            f"CI [{lo:.4f}, {hi:.4f}] straddles break-even {be:.4f}: "
            "the sign of the net effect is not identified."
        )

    ship_if_no_effect_total = cost * n_target
    forgo_gain_per_user = max(expected_net_per_user, 0.0)
    forgo_gain_total = forgo_gain_per_user * n_target

    return {
        "decision": decision,
        "target_segment": target,
        "break_even_uplift": be,
        "observed_uplift": observed,
        "ci": [lo, hi],
        "ci_level": ci_level,
        "n_target": n_target,
        "expected_net_per_user": expected_net_per_user,
        "cost_of_error": {
            "ship_if_no_effect": {"per_user": cost, "total": ship_if_no_effect_total},
            "no_ship_if_effect": {"per_user": forgo_gain_per_user, "total": forgo_gain_total},
            "worst_case_total": max(ship_if_no_effect_total, forgo_gain_total),
        },
        "rationale": rationale,
    }
