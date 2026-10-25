"""Tests for the pre-registered ship / no_ship / inconclusive rule."""
from __future__ import annotations

import pathlib
import re

import pytest

from src import config as C
from src.decision import INCONCLUSIVE, NO_SHIP, SHIP, break_even_uplift, ship_decision

ROOT = pathlib.Path(__file__).parent.parent


def _seg(value: float, lo: float, hi: float, n_t: int = 3000, n_c: int = 3000) -> dict:
    return {"value": value, "lo": lo, "hi": hi, "n_t": n_t, "n_c": n_c, "n_total": n_t + n_c}


def test_break_even_is_cost_over_value():
    assert break_even_uplift(40, 2) == 0.05


def test_break_even_rejects_non_positive_value():
    with pytest.raises(ValueError, match="value_per_conversion"):
        break_even_uplift(0, 2)


def test_ship_when_lower_ci_clears_break_even():
    r = ship_decision({"new": _seg(0.11, 0.09, 0.13)})
    assert r["decision"] == SHIP == "ship"


def test_no_ship_when_upper_ci_below_break_even():
    r = ship_decision({"new": _seg(0.01, -0.001, 0.027)})
    assert r["decision"] == NO_SHIP == "no_ship"


def test_inconclusive_when_ci_straddles_break_even():
    r = ship_decision({"new": _seg(0.04, -0.01, 0.08)})
    assert r["decision"] == INCONCLUSIVE == "inconclusive"


def test_cost_of_error_uses_the_real_numbers():
    n = 6000
    seg = _seg(0.1108, 0.0894, 0.1357, n_t=n // 2, n_c=n // 2)
    r = ship_decision({"new": seg})
    assert r["n_target"] == n
    assert r["decision"] == SHIP
    assert r["break_even_uplift"] == pytest.approx(0.05)
    assert r["expected_net_per_user"] == pytest.approx(0.1108 * 40 - 2)

    err = r["cost_of_error"]
    assert err["ship_if_no_effect"]["per_user"] == pytest.approx(2.0)
    assert err["ship_if_no_effect"]["total"] == pytest.approx(2 * n)
    assert err["no_ship_if_effect"]["total"] == pytest.approx((0.1108 * 40 - 2) * n)
    assert err["worst_case_total"] == pytest.approx((0.1108 * 40 - 2) * n)


def test_missing_target_segment_raises():
    with pytest.raises(ValueError, match="new"):
        ship_decision({"returning": _seg(0.01, 0.0, 0.02)})


def test_decision_config_matches_the_documented_constants():
    assert C.DECISION["target_segment"] == "new"
    assert C.DECISION["value_per_conversion"] == 40.0
    assert C.DECISION["campaign_cost_per_user"] == 2.0
    assert C.DECISION["ci_level"] == 0.95


def test_prereg_doc_is_bound_to_the_code():
    """The document and the rule must agree on every literal that changes it.

    Parsing the constants (not just checking a substring like "2") is what makes
    an edit to either side fail: change the config and the doc no longer matches,
    edit the rule in the doc and the config no longer matches.
    """
    text = (ROOT / "docs" / "preregistration.md").read_text(encoding="utf-8")
    for key in ("value_per_conversion", "campaign_cost_per_user", "ci_level"):
        match = re.search(rf"{key}`?\s*=\s*`?([0-9]+(?:\.[0-9]+)?)", text)
        assert match, f"preregistration.md does not state {key}"
        assert float(match.group(1)) == C.DECISION[key], key
    assert "`target_segment = new`" in text

    be = break_even_uplift(C.DECISION["value_per_conversion"], C.DECISION["campaign_cost_per_user"])
    assert f"break_even = {be:.2f}" in text

    # §3: every branch of the rule is bound to its verdict. Reading the line that
    # states the condition (not just "the word appears somewhere") means deleting
    # or swapping a branch fails the test.
    for condition, verdict in (
        ("`lo > break_even`", "ship"),
        ("`hi < break_even`", "no_ship"),
        ("CI накрывает break_even", "inconclusive"),
    ):
        line = next((ln for ln in text.splitlines() if condition in ln), None)
        assert line is not None, f"preregistration.md does not state the {condition} branch"
        assert re.search(rf"\*\*{verdict}\*\*", line), f"{condition} must map to {verdict}"

    # §4: the cost-of-error formulas the rule is scored by, verbatim.
    for formula in (
        "`campaign_cost_per_user * n_target`",
        "`max(observed * value - cost, 0) * n_target`",
        "максимум из двух веток",
    ):
        assert formula in text, f"preregistration.md does not bind the error cost: {formula}"
